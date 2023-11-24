from server import IntentRequest, IntentLog, db

def test_invalid_request(client):
    response = client.post('/intent', json={})
    assert response.status_code == 400
    assert 'Request body is missing.' in response.json['message']
def test_new_intent_request(app):
    with app.app_context():
        intent_request = IntentRequest(request_text='Test text', response_data='Test response', response_time=1.23)
        db.session.add(intent_request)
        db.session.commit()
        assert intent_request.id is not None
def test_example_endpoint(client):
    response = client.get('/api/example')
    assert response.status_code == 200
    assert response.json == {"message": "Success"}

