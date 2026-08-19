@echo off
REM ============================================================
REM Daryn — Автоматическая настройка среды (Windows)
REM Запустить один раз от имени обычного пользователя
REM ============================================================

echo.
echo  ██████╗  █████╗ ██████╗ ██╗   ██╗███╗   ██╗
echo  ██╔══██╗██╔══██╗██╔══██╗╚██╗ ██╔╝████╗  ██║
echo  ██║  ██║███████║██████╔╝ ╚████╔╝ ██╔██╗ ██║
echo  ██║  ██║██╔══██║██╔══██╗  ╚██╔╝  ██║╚██╗██║
echo  ██████╔╝██║  ██║██║  ██║   ██║   ██║ ╚████║
echo  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═══╝
echo.
echo  Агро-Разведчик :: Настройка среды
echo  ============================================
echo.

REM --- Шаг 1: Создание виртуального окружения ---
echo [1/4] Создание виртуального окружения Python...
python -m venv .venv
if %ERRORLEVEL% neq 0 (
    echo [ОШИБКА] Python не найден или недоступен. Установите Python 3.10+
    pause
    exit /b 1
)
echo        OK: .venv создан

REM --- Шаг 2: Активация venv ---
echo [2/4] Активация окружения...
call .venv\Scripts\activate.bat

REM --- Шаг 3: Установка зависимостей ---
echo [3/4] Установка зависимостей (может занять 2-5 минут)...
pip install --upgrade pip -q
pip install -r requirements.txt -q
if %ERRORLEVEL% neq 0 (
    echo [ОШИБКА] Не удалось установить зависимости. Проверь requirements.txt
    pause
    exit /b 1
)
echo        OK: Все библиотеки установлены

REM --- Шаг 4: Инструкция по GEE ---
echo [4/4] Финальный шаг — авторизация Google Earth Engine:
echo.
echo  ВАЖНО: Выполни следующую команду в этом терминале:
echo.
echo     earthengine authenticate
echo.
echo  Она откроет браузер для входа в Google-аккаунт.
echo  После входа скопируй токен обратно в терминал.
echo.
echo  Если ещё нет доступа — зарегистрируйся на:
echo  https://signup.earthengine.google.com/
echo.
echo  ============================================
echo  Среда готова! Для активации в будущем используй:
echo     .venv\Scripts\activate.bat
echo  Для запуска проекта:
echo     python main.py
echo  Для дашборда:
echo     streamlit run dashboard\app.py
echo  ============================================
echo.
pause
