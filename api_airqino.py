import requests

base_url = "https://airqino-api.magentalab.it"


def get_current_values(station_name):
    endpoint = f"/getCurrentValues/{station_name}"
    url = base_url + endpoint

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return None


def get_hourly_avg(station_name, dt_from_string, dt_to_string):
    endpoint = f"/getHourlyAvg/{station_name}/{dt_from_string}/{dt_to_string}"
    url = base_url + endpoint

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.text
        return data
    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return None


def get_range(station_name, dt_from_string, dt_to_string):
    endpoint = f"/getRange/{station_name}/{dt_from_string}/{dt_to_string}"
    url = base_url + endpoint

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return None


def get_session_info(project_name):
    endpoint = f"/getSessionInfo/{project_name}"
    url = base_url + endpoint

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return None


def get_single_day(station_name, dt_from_string):
    endpoint = f"/getSingleDay/{station_name}/{dt_from_string}"
    url = base_url + endpoint

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return None


def get_station_status(station_id):
    endpoint = f"/getStationStatus/{station_id}"
    url = base_url + endpoint

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return None


def get_stations(project_name):
    endpoint = f"/getStations/{project_name}"
    url = base_url + endpoint

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return None


def get_station_hourly_avg(station_id):
    endpoint = f"/v3/getStationHourlyAvg/{station_id}"
    url = base_url + endpoint

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return None