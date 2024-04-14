import pytest
import requests

# This fixture can be used if you need to perform setup tasks, not needed directly here but useful in larger contexts.
@pytest.fixture
def query_url():
    return "http://localhost:8000/query"

# Using the fixture to provide the URL, making it easy to modify in one place if needed.
def test_valid_question(query_url):
    # Sending the POST request to the specified URL with JSON body.
    response = requests.post(query_url, json={"text": "How many days of vacation can I have per year?"})
    
    # Asserting that the HTTP response status code is 200.
    assert response.status_code == 200, "Expected status code 200, but got {}".format(response.status_code)

    # Parsing JSON response once to avoid multiple calls to response.json()
    response_data = response.json()

    # Asserting the structure and data types of the response
    assert "response" in response_data, "Expected 'response' key in the JSON response"
    assert isinstance(response_data["response"], str), "Expected 'response' to be a string"
    assert '5' in response_data["response"] or 'five' in response_data["response"], "Expected '5' or 'five' in the response"
    assert "sources" in response_data, "Expected 'sources' key in the JSON response"
    assert isinstance(response_data["sources"], list), "Expected 'sources' to be a list"
    assert len(response_data["sources"]) > 0, "Expected at least one source in the response"
    filtered_sources = list(filter(lambda x: x["file"] == 'GPT - leave policy.pdf' and x["page"] == "1", response_data["sources"]))
    assert len(filtered_sources) != 0, "Expected GPT - leave policy.pdf in sources"


# now test invalid as wrong input query
def test_invalid_question(query_url):
    # Sending the POST request to the specified URL with JSON body.
    response = requests.post(query_url, json={"apple": "This is not a valid question."})
    
    # Asserting that the HTTP response status code is 400.
    assert response.status_code == 422, "Expected status code 422, but got {}".format(response.status_code)


# now test invalid as wrong input query
def test_empty_question(query_url):
    # Sending the POST request to the specified URL with JSON body.
    response = requests.post(query_url, json={"text": ""})
    
    # Asserting that the HTTP response status code is 400.
    assert response.status_code == 400, "Expected status code 400, but got {}".format(response.status_code)



# The tests can be run using a pytest command in the terminal.
