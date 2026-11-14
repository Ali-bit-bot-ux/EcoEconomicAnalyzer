import json
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from config.settings import GRAIN_BELT_REGIONS, TARGET_CROPS
from modules.gis_engine import DISTRICT_REGISTRY
from modules.space_cache import (
    SpaceCacheError, archive_ndvi_climatology, archive_smap_series, build_query,
    load_noaa_kp, load_space_cache, offline_archive_status, save_noaa_kp, save_space_cache,
)
from modules.space_segment import (
    CHANNEL_LABELS, SATELLITES, SAR_COLUMNS, assess_space_weather,
    build_data_cube, build_observation_map, fetch_noaa_kp,
    generate_tasking_request, prepare_optical_observations, prepare_sar_observations,
)


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_observations(polygon_json, start, end, orbit_pass):
    import ee
    from pipeline.gee_client import GEEClient
    try:
        return GEEClient().get_space_observations(json.loads(polygon_json), start, end, orbit_pass=orbit_pass)
    except ee.EEException as error:
        raise RuntimeError("Earth Engine request failed") from error


def _uploaded_frame(upload, kind, region_id, district_id, year):
    frame = pd.read_csv(upload)
    if not {"region_id", "district_id"}.issubset(frame.columns):
        raise ValueError("CSV requires region_id and district_id")
    frame = frame[(frame["region_id"] == region_id) & (frame["district_id"] == district_id)]
    frame = prepare_sar_observations(frame) if kind == "sar" else prepare_optical_observations(frame)
    return frame[frame["date"].dt.year == year]


def render_space_segment_tab(*, lang, region_id, crop_id, year, shock_percent):
    kk = lang == "kk"
    text = lambda kazakh, russian: kazakh if kk else russian
    st.markdown("""<style>
    .st-key-space_segment [data-testid="stWidgetLabel"] p,
    .st-key-space_segment [data-testid="stRadio"] label p {color: #cbd5e1 !important;}
    .st-key-space_segment [data-testid="stAlert"] p {color: #dbeafe !important;}
    </style>""", unsafe_allow_html=True)
    region, crop = GRAIN_BELT_REGIONS[region_id], TARGET_CROPS[crop_id]
    st.header(text("Ғарыштық сегмент", "Космический сегмент"))
    st.caption(f"{region['name_' + lang]} · {crop['name_' + lang]} · {year}")
    st.info(text(
        "SAR — жер бетінен кері шашырау, диэлектрлік өтімділік пен биомассаны тікелей өлшеу емес. Дақыл мен стресс сценарийі спутниктік бақылауларды өзгертпейді.",
        "SAR измеряет обратное рассеяние, а не непосредственно диэлектрическую проницаемость или биомассу. Культура и стресс-сценарий не изменяют спутниковые наблюдения.",
    ))
    districts = DISTRICT_REGISTRY[region_id]
    district = st.selectbox(text("Талдау ауданы (AOI)", "Район анализа (AOI)"), districts,
                            format_func=lambda item: item["name_" + lang], key=f"space_district_{region_id}")
    geometry = {"type": "Polygon", "coordinates": [district["polygon"]]}
    st.caption(text("AOI — тізілімдегі жуық аудан шекарасы; анықталған құрғақшылық ошағы емес.",
                    "AOI — приближённый контур района из реестра, не выделенный очаг засухи."))
    orbit_pass = st.selectbox(text("Орбита бағыты", "Направление орбиты"), ["DESCENDING", "ASCENDING"],
                              format_func=lambda item: text("Төмендейтін", "Нисходящая") if item == "DESCENDING" else text("Көтерілетін", "Восходящая"), key=f"space_pass_{lang}")
    start_date, end_date = f"{year}-04-01", f"{year}-10-01"
    query = build_query(region_id=region_id, district_id=district["id"], year=year,
                        orbit_pass=orbit_pass, start_date=start_date, end_date=end_date, geometry=geometry)
    query_key = json.dumps(query, sort_keys=True)
    st.caption(text("GEE: 1 сәуір – 1 қазан (соңғы күн кірмейді), ортақ орбита, 100 м кеңістіктік орташалау. GEE сәтсіз болса — сол сұрауға арналған жергілікті кэш.",
                    "GEE: 1 апреля – 1 октября (конец не включён), единая орбита, пространственное усреднение 100 м. При сбое GEE — локальный кэш для того же запроса."))
    cache_payload, cache_error = None, None
    try:
        cache_payload = load_space_cache(query)
    except SpaceCacheError:
        cache_error = True
    if cache_error:
        st.warning(text("Жергілікті кэш табылды, бірақ оқылмайды; деректер қолданылмайды.",
                        "Локальный кэш найден, но не читается; данные не используются."))
    cache_provider = cache_payload["provenance"]["provider"] if cache_payload else "Google Earth Engine"
    channel_options = list(CHANNEL_LABELS[lang])
    default_channel = "rvi" if cache_payload and cache_payload["observations"].get("sar") and not cache_payload["observations"].get("optical") else channel_options[0]
    channel = st.radio(text("Спутниктік арна", "Спутниковый канал"), channel_options,
                       format_func=lambda item: CHANNEL_LABELS[lang][item], horizontal=True,
                       index=channel_options.index(default_channel), key="space_channel")
    saved_key, observations = st.session_state.get("space_observations", (None, {}))
    source = "live" if saved_key == query_key else "cache" if cache_payload else "none"
    if source == "cache":
        observations = cache_payload["observations"]
    elif source == "none":
        observations = {}
    fetch_clicked = st.button(text("GEE бақылауларын жүктеу / жаңарту", "Загрузить / обновить наблюдения GEE"), key="space_fetch")
    if fetch_clicked:
        try:
            with st.spinner(text("Sentinel-1/2 деректері сұралуда…", "Запрос данных Sentinel-1/2…")):
                observations = _fetch_observations(json.dumps(district["polygon"]), start_date, end_date, orbit_pass)
            save_space_cache(query, observations)
            st.session_state["space_observations"] = (query_key, observations)
            source = "live"
            st.success(text("GEE live-деректері жүктелді және оффлайн-кэшке сақталды.",
                            "Живые данные GEE загружены и сохранены в оффлайн-кэш."))
        except (ImportError, RuntimeError, requests.RequestException) as error:
            if cache_payload:
                observations = cache_payload["observations"]
                source = "cache"
                st.warning(text("GEE қолжетімсіз. Презентация үшін жергілікті кэш қолданылады; оның уақыты жоғарыда көрсетілген.",
                                "GEE недоступен. Для презентации используется локальный кэш; его время указано выше."))
            else:
                st.error(text("GEE қолжетімсіз және бұл сұрауға жергілікті кэш жоқ. CSV импортын қолданыңыз; жасанды деректер қолданылмайды.",
                              "GEE недоступен, и локального кэша для этого запроса нет. Используйте CSV; подмена данных не выполняется."))
            st.caption(type(error).__name__)
    if source == "cache" and not fetch_clicked:
        st.info(text(f"Оффлайн-кэш · {cache_provider} · {cache_payload['cached_at']} · SAR {len(observations.get('sar', []))} · NDVI {len(observations.get('optical', []))}. Деректер ағымдағы деп белгіленбейді.",
                     f"Оффлайн-кэш · {cache_provider} · {cache_payload['cached_at']} · SAR {len(observations.get('sar', []))} · NDVI {len(observations.get('optical', []))}. Данные не выдаются за текущие."))
    with st.expander(text("Өз бақылауларыңызды импорттау (CSV)", "Импорт собственных наблюдений (CSV)")):
        st.caption(text("Әр жолда region_id және district_id болуы керек. Тек таңдалған аудан мен маусым көрсетіледі. Дереккөзді пайдаланушы растайды.",
                        "Каждая строка должна содержать region_id и district_id. Показывается только выбранный район и сезон. Источник подтверждает пользователь."))
        st.code("region_id=" + region_id + "; district_id=" + district["id"])
        st.code("SAR: region_id,district_id," + ",".join(SAR_COLUMNS) + "\nNDVI: region_id,district_id,date,scene_id,ndvi")
        sar_upload = st.file_uploader(text("SAR бақылаулары CSV", "Наблюдения SAR CSV"), type=["csv"], key="space_sar_upload")
        optical_upload = st.file_uploader(text("NDVI бақылаулары CSV", "Наблюдения NDVI CSV"), type=["csv"], key="space_ndvi_upload")
    sar = pd.DataFrame(columns=[*SAR_COLUMNS, "rvi", "psi_radar", "vh_vv_ratio"])
    optical = pd.DataFrame(columns=["date", "scene_id", "ndvi"])
    for kind, upload in (("sar", sar_upload), ("optical", optical_upload)):
        try:
            if upload is not None:
                upload.seek(0)
                frame = _uploaded_frame(upload, kind, region_id, district["id"], year)
                st.caption(text("Дереккөз: пайдаланушы CSV; түпнұсқалығы автоматты тексерілмеген.",
                                "Источник: пользовательский CSV; подлинность автоматически не проверена."))
            elif observations.get(kind):
                raw = pd.DataFrame(observations[kind])
                frame = prepare_sar_observations(raw) if kind == "sar" else prepare_optical_observations(raw)
                label = "Google Earth Engine" if source == "live" else text("Жергілікті кэш", "Локальный кэш") + " · " + cache_provider
                st.caption(label + " · " + ("COPERNICUS/S1_GRD" if kind == "sar" else "COPERNICUS/S2_SR_HARMONIZED"))
            else:
                continue
            if kind == "sar":
                sar = frame[frame["orbit_pass"] == orbit_pass]
            else:
                optical = frame
        except (ValueError, TypeError, UnicodeError, pd.errors.ParserError):
            st.error(text("CSV/бақылау жарамсыз: бағандарды, күнді, орбитаны және сандық мәндерді тексеріңіз.",
                          "Некорректный CSV/наблюдение: проверьте столбцы, дату, орбиту и числовые значения."))
    if not sar.empty:
        relative = st.selectbox(text("Салыстырмалы орбита", "Относительная орбита"), sorted(sar["relative_orbit"].unique()), key="space_relative")
        sar = sar[sar["relative_orbit"] == relative]
    selected = optical if channel == "ndvi" else sar
    value = None if selected.empty else float(selected.iloc[-1][channel])
    archive_ndvi = archive_ndvi_climatology(region_id) if channel == "ndvi" else None
    if value is None:
        st.warning(text("Таңдалған арна, аудан және маусым үшін бақылаулар жоқ. GEE немесе CSV арқылы жүктеңіз.",
                        "Нет наблюдений для выбранного канала, района и сезона. Загрузите их через GEE или CSV."))
        if archive_ndvi:
            st.caption(text("Төмендегі қисық — 2015–2023 жылдардағы СҚО климатологиясы, таңдалған жылдың өлшемі емес.",
                            "Кривая ниже — климатология СКО 2015–2023, не измерение выбранного года."))
    else:
        unit = " дБ" if channel.endswith("_db") else ""
        st.metric(CHANNEL_LABELS[lang][channel], f"{value:.3f}{unit}")
        st.caption(f"UTC: {selected.iloc[-1]['date'].isoformat()} · {selected.iloc[-1]['scene_id']}")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(build_observation_map(geometry, channel, value, lang=lang), use_container_width=True, theme=None, key="space_map")
        st.caption(text("Түс — AOI-дің соңғы орташа мәні; пиксельдік аномалия картасы емес. Бақылаусыз контур сұр.",
                        "Цвет — последнее среднее по AOI, не пиксельная карта аномалий. Без наблюдений контур серый."))
    with right:
        fig = go.Figure()
        if not selected.empty:
            fig.add_trace(go.Scatter(x=selected["date"], y=selected[channel], mode="lines+markers", name=channel, connectgaps=False))
        elif archive_ndvi:
            fig.add_trace(go.Scatter(x=archive_ndvi["frame"]["doy"], y=archive_ndvi["frame"]["ndvi_mean"],
                                     mode="lines", name=archive_ndvi["label_" + lang],
                                     line={"dash": "dash", "color": "#8a9bb0"}))
        fig.update_layout(height=340, template="plotly_dark", title=CHANNEL_LABELS[lang][channel],
                          xaxis_title=text("Күн (DOY) / бақылау күні", "День (DOY) / дата наблюдения"),
                          yaxis_title="дБ" if channel.endswith("_db") else channel)
        st.plotly_chart(fig, use_container_width=True, theme=None, key="space_series")
        if not selected.empty and archive_ndvi:
            st.caption(text("Жедел қисық бар; архив климатологиясы салыстыру ретінде қолжетімді.",
                            "Есть оперативный ряд; архивная климатология доступна для сравнения."))
    st.latex(r"\sigma^0_{lin}=10^{\sigma^0_{dB}/10},\qquad RVI_{dual}=\frac{4\sigma^0_{VH,lin}}{\sigma^0_{VV,lin}+\sigma^0_{VH,lin}},\qquad PSI_{radar}=\frac{\sigma^0_{VH,lin}}{\sigma^0_{VV,lin}+\sigma^0_{VH,lin}}")
    st.caption(text("RVI — dual-pol индексі, диапазоны [0, 4]. PSI_radar — research-жұмысындағы Plant Structure Index, диапазоны [0, 1]; RVI = 4 × PSI_radar. Екеуі де AOI орташа қуаттарынан есептеледі, пиксельдік индекстерді орташалау емес. VV/VH-ға кедір-бұдырлық, түсу бұрышы және өсімдік құрылымы да әсер етеді.",
                    "RVI — dual-pol индекс, диапазон [0, 4]. PSI_radar — Plant Structure Index из research-работы, диапазон [0, 1]; RVI = 4 × PSI_radar. Оба считаются из средних мощностей AOI, а не усреднением пиксельных индексов. VV/VH зависят также от шероховатости, угла падения и структуры растительности."))
    if not sar.empty:
        depol = go.Figure(go.Scatter(x=sar["date"], y=sar["vh_vv_ratio"], mode="lines+markers", name="VH/VV"))
        depol.update_layout(height=230, template="plotly_dark", title=text("Деполяризация динамикасы: VH/VV (сызықтық қуат)", "Динамика деполяризации: VH/VV (линейная мощность)"))
        st.plotly_chart(depol, use_container_width=True, theme=None, key="space_depol")
    _render_tasking(geometry, lang, region_id, district["id"], crop_id, year, shock_percent)
    _render_weather(lang, region_id)
    _render_offline_archive(lang)
    st.subheader(text("3D Ғарыш–Жер деректер текшесі", "3D Куб данных Космос–Земля"))
    st.caption(text("Биіктік масштабы шартты. SAR сәулесі тамырға жетпейді: тамыр аймағы су балансы моделімен бағаланады. SMAP L3 — беткі ылғал; SMAP L4 — модельдік тамыр аймағы (0–1 м).", 
                    "Масштаб высот условный. SAR не зондирует корни: корневая зона оценивается моделью водного баланса. SMAP L3 — поверхностная влага; SMAP L4 — модельная корневая зона (0–1 м)."))
    root_range = crop["root_depth_meters"]
    root_depth = float(root_range.split(" - ")[0])
    clouds = st.slider(text("Бұлттылық сценарийі (%)", "Сценарий облачности (%)"), 0, 100, 70, key="space_clouds")
    st.caption(text(f"Дақыл тамырының анықтамалық аралығы: {root_range} м; сызбада төменгі шек көрсетілген.",
                    f"Справочный диапазон корней культуры: {root_range} м; на схеме показана нижняя граница."))
    st.plotly_chart(build_data_cube(lang=lang, root_depth_m=root_depth, cloud_cover=clouds, channel=channel, value=value),
                    use_container_width=True, theme=None, key="space_cube")
    with st.expander(text("Бастапқы дереккөздер және шектеулер", "Первоисточники и ограничения")):
        st.markdown("[Sentinel-1 GRD · GEE](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD)")
        st.markdown("[NASA DEVELOP · dual-pol RVI](https://ntrs.nasa.gov/api/citations/20250003089/downloads/2025Spring_PUP_GulfCoastEco_FD-Final.pdf)")
        st.markdown("[Қазақстан Ғарыш Сапары · KazEOSat](https://www.gharysh.kz/infrastruktura/AboutKA_DZZ22_jnot/)")
        st.markdown("[ESA · Ionospheric Delay](https://gssc.esa.int/navipedia/index.php/Ionospheric_Delay)")
        st.markdown("[NASA/JPL · Faraday](https://descanso.jpl.nasa.gov/propagation/1108/1108Chapter2.pdf)")
        st.markdown("[NOAA · Kp](https://www.spaceweather.gov/products/planetary-k-index)")
        st.caption(text("Оператор KazEOSat-1 үшін 750 км көрсетеді (бастапқы ТЗ-дегі 675 км расталмады). Dual-pol RVI толық поляриметриялық RVI-ге тең емес. TEC спутникке дейінгі сәуле бөлігін қамтуы керек, толық GNSS бағанын емес.",
                        "Оператор указывает для KazEOSat-1 750 км (675 км из исходного ТЗ не подтверждены). Dual-pol RVI не эквивалентен полнополяриметрическому RVI. TEC должен соответствовать пути до спутника, а не всему GNSS-столбу."))


def _render_tasking(geometry, lang, region_id, district_id, crop_id, year, shock_percent):
    kk = lang == "kk"
    text = lambda kazakh, russian: kazakh if kk else russian
    st.subheader("Қазақстан Ғарыш Сапары · KazEOSat")
    for column, (name, spec) in zip(st.columns(2), SATELLITES.items()):
        with column:
            st.metric(name, f"{spec['orbit_altitude_km']} км", text("Жобалық орбита", "Проектная орбита"), delta_color="off")
            st.write(text("Ажыратылым", "Разрешение") + f": {spec['resolution_m']:g} м · " + text("Қамту жолағы", "Полоса захвата") + f": {spec['swath_km']} км")
            st.caption(text("Мультиспектр", "Мультиспектр") + f": {spec['multispectral_resolution_m']:g} м · " + ", ".join(spec["bands"]))
    st.warning(text("Бұл — КА паспорты, ағымдағы телеметрия емес. 1 м — субметрлік емес. Операторға қосылу, ұшу терезесі және бұлт бойынша растау жоқ.",
                    "Это паспортные характеристики, не текущая телеметрия. 1 м — не субметровое разрешение. Подключение к оператору, окно пролёта и подтверждение облачности отсутствуют."))
    satellite = st.selectbox(text("Нысаналы КА", "Целевой КА"), list(SATELLITES), key="space_satellite")
    distance = st.number_input(text("Жерүсті трассадан AOI-ге дейінгі қашықтық, км (міндетті емес)",
                                    "Расстояние от подспутниковой точки до AOI, км (необязательно)"), min_value=0.0, max_value=2000.0, value=None, key="space_distance")
    st.caption(text("Қашықтық берілсе, сфералық Жер үшін надирден ауытқу бағаланады. Бұл нақты крен емес; кренге эфемерида және қозғалыс бағыты қажет.",
                    "По введённому расстоянию оценивается отклонение от надира для сферической Земли. Это не командный крен: для него нужны эфемериды и направление движения."))
    smai = st.number_input(text("Аномалия SMAI (міндетті емес, < −1.5)", "Аномалия SMAI (необязательно, < −1.5)"), value=None, key="space_smai")
    source = st.text_input(text("SMAI дереккөзі / өнім идентификаторы", "Источник SMAI / идентификатор продукта"), key="space_smai_source")
    today_utc = datetime.now(timezone.utc).date()
    observed = st.date_input(text("SMAI бақылау күні (UTC)", "Дата наблюдения SMAI (UTC)"), value=today_utc, max_value=today_utc, key="space_smai_date")
    st.caption(text("SMAI енгізілмесе, тек қолмен жоспарлау сұрауы жасалады. Қазіргі AOI-ді оператор нақтылауы тиіс; бұл автоматты сегменттелген ошақ емес.",
                    "Без SMAI создаётся запрос ручного планирования. AOI требует уточнения оператором: это не автоматически сегментированный очаг."))
    context = {"region_id": region_id, "district_id": district_id, "crop_id": crop_id, "analysis_year": year,
               "scenario_moisture_drop_percent": shock_percent, "aoi_origin": "approximate_district_registry",
               "scenario_is_observation": False}
    params = {"geometry": geometry, "satellite": satellite, "smai": smai, "anomaly_source": source,
              "observed_at": datetime.combine(observed, datetime.min.time(), tzinfo=timezone.utc).isoformat(),
              "ground_distance_km": distance, "context": context}
    fingerprint = json.dumps(params, sort_keys=True)
    if st.button(text("KazEOSat ұшу тапсырмасын қалыптастыру", "Сформировать полётное задание KazEOSat"), key="space_tasking"):
        try:
            request = generate_tasking_request(**params)
            st.session_state["space_tasking_result"] = (fingerprint, request)
        except (ValueError, TypeError):
            st.session_state.pop("space_tasking_result", None)
            st.error(text("Сұрау жарамсыз: SMAI < −1.5, дереккөз, бақылау күні және геометриялық көрінуді тексеріңіз.",
                          "Некорректный запрос: проверьте SMAI < −1.5, источник, дату наблюдения и геометрическую видимость."))
    saved_fingerprint, request = st.session_state.get("space_tasking_result", (None, None))
    if saved_fingerprint == fingerprint and request:
        st.success(text("Жергілікті жоба дайын. Операторға жіберілген жоқ.", "Локальный черновик готов. Оператору не отправлен."))
        if request["planning"]["estimated_off_nadir_deg"] is not None:
            st.metric(text("Надирден ауытқу бағасы", "Оценка отклонения от надира"), f"{request['planning']['estimated_off_nadir_deg']:.2f}°")
        st.download_button(text("Тапсырманы жүктеу (.json)", "Скачать задание (.json)"),
                           data=json.dumps(request, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"),
                           file_name=f"AgriCascade_{satellite}_{district_id}_tasking.json", mime="application/json", key="space_download")
        with st.expander(text("Сұрау метадеректері", "Метаданные запроса")):
            st.json(request)


def _render_offline_archive(lang):
    kk = lang == "kk"
    text = lambda kazakh, russian: kazakh if kk else russian
    with st.expander(text("Оффлайн-архив data/processed", "Оффлайн-архив data/processed")):
        files = offline_archive_status()
        st.caption(text(f"{len(files)} parquet файл сақталған. Олар өзгертілмейді; GEE кэші бөлек `space_cache/` ішінде жатыр.",
                        f"Сохранено {len(files)} parquet-файлов. Они не изменяются; кэш GEE хранится отдельно в `space_cache/`."))
        if files:
            st.dataframe(pd.DataFrame(files), use_container_width=True, hide_index=True)
        st.caption(text("VV/VH мұрағаты тек сәтті GEE/CSV жүктеуінен кейін пайда болады; ойдан жасалған радар деректері жоқ.",
                        "Архив VV/VH появится только после успешной загрузки GEE/CSV; синтетических радарных данных нет."))


def _render_weather(lang, region_id):
    kk = lang == "kk"
    text = lambda kazakh, russian: kazakh if kk else russian
    st.subheader(text("Ғарыштық ауа райы · SMAP L-band", "Космическая погода · SMAP L-band"))
    if st.button(text("NOAA Kp жаңарту", "Обновить NOAA Kp"), key="space_weather_fetch"):
        try:
            live_kp = fetch_noaa_kp()
            save_noaa_kp(live_kp)
            st.session_state["space_kp"] = live_kp
            st.session_state["space_kp_source"] = "live"
            st.session_state.pop("space_kp_error", None)
        except (requests.RequestException, ValueError, TypeError):
            st.session_state["space_kp_error"] = True
    if st.session_state.get("space_kp_error"):
        st.warning(text("NOAA жаңартылмады. Соңғы сәтті мән болса, оның уақыты төменде көрсетілген.",
                        "NOAA не обновлён. Если есть последнее успешное значение, ниже указано его время."))
    observation = st.session_state.get("space_kp")
    if not observation:
        try:
            observation = load_noaa_kp()
            if observation:
                st.session_state["space_kp"] = observation
                st.session_state["space_kp_source"] = "cache"
        except SpaceCacheError:
            observation = None
    kp = None
    if observation:
        age_hours = (datetime.now(timezone.utc) - datetime.fromisoformat(observation["observed_at"])).total_seconds() / 3600
        st.metric("NOAA Kp", f"{observation['kp']:.2f}")
        source_note = text(" · жергілікті кэш", " · локальный кэш") if st.session_state.get("space_kp_source") == "cache" else ""
        st.caption(f"UTC: {observation['observed_at']} · NOAA SWPC · " + text("3 сағаттық бағалау", "3-часовая оценка") + source_note)
        if age_hours > 6 or st.session_state.get("space_kp_error"):
            st.warning(text("Сақталған Kp ағымдағы сапаны растамайды.", "Сохранённый Kp не подтверждает текущее качество."))
        else:
            kp = observation["kp"]
    else:
        st.info(text("Kp әлі жүктелмеген; тұрақты мәнмен алмастырылмайды.", "Kp ещё не загружен; фиксированное значение не подставляется."))
    basin_by_region = {"sko": "tobol", "kostanay": "tobol"}
    smap_archive = archive_smap_series(basin_by_region.get(region_id))
    if smap_archive:
        latest = smap_archive["latest"]
        st.metric(text("SMAP L3 · Тобыл бассейні (архив)", "SMAP L3 · бассейн Тобол (архив)"),
                  f"{latest['soil_moisture']:.3f} м³/м³")
        st.caption(text(f"Соңғы өлшем: {latest['date']} · data/processed/smap_timeseries.parquet. Бұл ағымдағы дала өлшемі емес.",
                        f"Последнее измерение: {latest['date']} · data/processed/smap_timeseries.parquet. Это не текущий полевой замер."))
        with st.expander(text("SMAP архив қисығы", "Архивная кривая SMAP")):
            st.line_chart(smap_archive["frame"].set_index("date")["soil_moisture"], height=180)
    tec = st.number_input(text("Өлшенген көлбеу TEC (STEC), TECU", "Измеренный наклонный TEC (STEC), TECU"),
                          min_value=0.0, max_value=1000.0, value=None, key="space_tec")
    field = st.number_input(text("Сәуле бойындағы электрондармен өлшенген орташа B∥, мкТл", "Среднее B∥ вдоль луча, взвешенное по электронам, мкТл"),
                            min_value=-100.0, max_value=100.0, value=None, key="space_bparallel")
    st.caption(text("TEC автоматты жүктелмейді: GNSS/IONEX өлшемі қажет. VTEC-ті STEC-ке геометриялық түрлендірусіз қолданбаңыз. Kp арқылы TEC есептелмейді. Енгізілген мәндер ағымдағы спутниктік өлшем деп расталмайды.",
                    "TEC не загружается автоматически: нужен замер GNSS/IONEX. Нельзя подставлять VTEC без преобразования геометрии в STEC. TEC не выводится из Kp. Введённые значения не подтверждаются как текущие спутниковые измерения."))
    weather = assess_space_weather(kp=kp, tec_tecu=tec, b_parallel_ut=field)
    status = {
        "unknown": text("Геомагниттік фон белгісіз", "Геомагнитный фон неизвестен"),
        "quiet": text("Геомагниттік фон тыныш", "Геомагнитный фон спокойный"),
        "unsettled": text("Геомагниттік белсенділік жоғарылаған", "Геомагнитная активность повышена"),
        "storm": text("Геомагниттік дауыл: Kp ≥ 5", "Геомагнитная буря: Kp ≥ 5"),
    }[weather["geomagnetic_status"]]
    (st.warning if weather["geomagnetic_status"] == "storm" else st.info)(status)
    if tec is not None:
        st.metric(text("Ионосфералық топтық кідіріс, м", "Ионосферная групповая задержка, м"), f"{weather['group_delay_m']:.3f}")
    if weather["faraday_rotation_deg"] is not None:
        st.metric(text("Фарадей бұрылуының бағасы", "Оценка фарадеевского вращения"), f"{weather['faraday_rotation_deg']:.3f}°")
    else:
        st.info(text("Фарадей бұрылуына STEC және сәуле бойындағы B∥ қажет.", "Для фарадеевского вращения нужны STEC и B∥ вдоль луча."))
    st.latex(r"\Delta L=40.3\,STEC/f^2,\quad \Omega\approx2.36\times10^4\,STEC\,\langle B_{\parallel}\rangle/f^2")
    st.caption(text("f = 1.41 ГГц; 1 TECU = 10¹⁶ электрон/м²; B — тесла; Ω — радиан. Бұл бағалау SMAP түзетуін орындамайды және ылғал өлшемін валидацияламайды. Өнімнің QA жалаушалары қажет; сапа пайызы ойдан шығарылмайды.",
                    "f = 1.41 ГГц; 1 TECU = 10¹⁶ электрон/м²; B — тесла; Ω — радианы. Оценка не выполняет коррекцию SMAP и не валидирует измерение влажности. Нужны QA-флаги продукта; процент качества не выдумывается."))
