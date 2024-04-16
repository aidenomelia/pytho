import requests

def get_weather(city):
    api_key = "a90f9c8cb1f37200a406f067795cba62"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    
    response = requests.get(url)
    data = response.json()
    
    if "cod" in data and data["cod"] == "404":
        return "City not found. Please try again."
    
    try:
        weather_description = data["weather"][0]["description"]
        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]
        
        weather_report = f"Weather in {city}: {weather_description}\nTemperature: {temperature}°C\nHumidity: {humidity}%\nWind Speed: {wind_speed} m/s"
        return weather_report
    except KeyError:
        return "Unable to retrieve weather data. Please try again later."


def main():
    city = input("Enter city name: ")
    print(get_weather(city))

if __name__ == "__main__":
    main()
