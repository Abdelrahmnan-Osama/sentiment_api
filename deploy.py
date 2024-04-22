from flask_restful import Api, Resource
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS module

from model import sentiment_score

# Create the API
app = Flask(__name__)
api = Api(app)

# Enable CORS for all domains
CORS(app)

class Prediction(Resource): 
    def post(self):
        # Retrieve data from request
        data = request.get_json()
        
        # Perform sentiment analysis for each comment
        predictions = []
        for item in data:
            comment = item['comment']
            rating = sentiment_score(comment)
            predictions.append({
                'Id': item['Id'],
                'comment': comment,
                'rating': str(rating)
            })
        
        return jsonify(predictions)

api.add_resource(Prediction, '/prediction')

if __name__ == '__main__':
    app.run(debug=True)