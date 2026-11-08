import ee

print("Запускаем авторизацию напрямую через код...")

try:
    # 1. Запускаем окно авторизации (выдаст ссылку в терминал)
    ee.Authenticate(auth_mode='notebook')
    
    # 2. Сразу инициализируемся с указанием проекта
    ee.Initialize(project='daryn-505110')
    
    print("✅ ЖБ вариант сработал! Доступ получен, токен сохранен.")
except Exception as e:
    print(f"❌ Ошибка: {e}")

