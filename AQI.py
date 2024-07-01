def calculate_aqi(C, breakpoints):
    for (C_low, C_high, I_low, I_high) in breakpoints:
        if C_low <= C <= C_high:
            AQI = ((I_high - I_low) / (C_high - C_low)) * (C - C_low) + I_low
            return round(AQI)
    return None

def get_aqi_category(aqi):
    if aqi is None:
        return "Fuori scala", "N/A", 100
    elif 0 <= aqi <= 50:
        return "Buona", "Verde", 0
    elif 51 <= aqi <= 100:
        return "Moderata", "Giallo", 1
    elif 101 <= aqi <= 150:
        return "Non salutare per gruppi sensibili", "Arancione", 2
    elif 151 <= aqi <= 200:
        return "Non salutare", "Rosso", 3
    elif 201 <= aqi <= 300:
        return "Molto non salutare", "Viola", 4
    elif 301 <= aqi <= 500:
        return "Pericolosa", "Marrone", 5


def aqi_pm10(val):
    if 0 <= val <= 20:
        return "Good", "Verde", 0
    elif 20 < val <= 40:
        return "Fair", "Giallo", 1
    elif 40 < val <= 50:
        return "Moderate", "Arancione", 2
    elif 50 < val <= 100:
        return "Poor", "Rosso", 3
    elif 100 < val <= 150:
        return "Very poor", "Viola", 4
    elif 150 <= val <= 1200:
        return "Extremely poor", "Marrone", 5
    else:
        return "N/A", "N/A", 100


def aqi_pm25(val):
    if 0 <= val <= 10:
        return "Good", "Verde", 0
    elif 10 < val <= 20:
        return "Fair", "Giallo", 1
    elif 20 < val <= 25:
        return "Moderate", "Arancione", 2
    elif 25 < val <= 50:
        return "Poor", "Rosso", 3
    elif 50 < val <= 75:
        return "Very poor", "Viola", 4
    elif 75 <= val <= 800:
        return "Extremely poor", "Marrone", 5
    else:
        return "N/A", "N/A", 100

# Breakpoints for PM2.5 and PM10
pm25_breakpoints = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500)
]

pm10_breakpoints = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 604, 301, 500)
]
