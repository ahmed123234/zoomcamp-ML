import requests

url = "http://127.0.0.1:9696/predict"

client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}

# Send the POST request
response = requests.post(url, json=client)
result = response.json()

# Extract and print the probability
probability = result.get("conversion_probability")

print(f"Client: {client}")
print(f"Probability of subscription: {probability:.3f}")


