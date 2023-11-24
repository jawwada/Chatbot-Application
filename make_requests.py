import requests
import random


def make_requests(url, num_requests):
    # Sample texts for making requests
    sample_texts = [
        "find me a flight to Miami",
        "what's the weather like in New York?",
        "play some rock music",
        "set an alarm for 7 AM",
        "how to make a pancake?",
        # Add more varied texts as needed
    ]

    for i in range(num_requests):
        # Randomly pick a text from the sample texts
        text = random.choice(sample_texts)
        payload = {"text": text}

        # Make the POST request with JSON payload
        response = requests.post(url, json=payload)
        print(f"Request {i + 1}: Status Code {response.status_code}, Response: {response.json()}")


if __name__ == "__main__":
    # URL of your Flask `/intent` endpoint
    url = 'http://localhost:8090/intent'  # Adjust the port if different

    # Number of requests to make
    num_requests = 100

    make_requests(url, num_requests)
