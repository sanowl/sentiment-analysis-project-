# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertTokenizer
from model import SentimentModel
from config import CONFIG

app = Flask(__name__)
CORS(app)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = SentimentModel(
    tokenizer.vocab_size, CONFIG['embedding_dim'], CONFIG['hidden_dim'], CONFIG['output_dim']
)
model.load_state_dict(torch.load('model.pth'))
model.eval()

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    text = request.json['text']
    encoding = tokenizer(text, truncation=True, padding=True, return_tensors='pt')
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    output = model(input_ids, attention_mask).squeeze(0)
    sentiment = torch.sigmoid(output).item()
    sentiment_label = 'Positive' if sentiment >= 0.5 else 'Negative'
    return jsonify({'sentiment': sentiment_label})

if __name__ == '__main__':
    app.run(debug=True)
