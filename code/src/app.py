import spacy
import pandas as pd
from transformers import pipeline
from sklearn.ensemble import IsolationForest
from fastapi import FastAPI, HTTPException
import json
import numpy as np
import csv
import re
import uvicorn
import logging

import Entity_Classifier as ec
import requests
from io import StringIO
from typing import Dict, List
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ENTITY_CSV_PATH = "sanctions.csv"
sanctions_df = pd.DataFrame()
# Load NLP models with error handling
try:
    nlp = spacy.load("en_core_web_sm")
    ner_model = pipeline("ner", model="distilbert-base-cased")
    generator = pipeline("text-generation", model="facebook/opt-125m")
    sentiment_analyzer = pipeline("sentiment-analysis", model="allenai/longformer-base-4096")
    sanctions_df = pd.read_csv(ENTITY_CSV_PATH, quotechar='"', skipinitialspace=True)
    sanctions_df.columns = sanctions_df.columns.str.strip()
except Exception as e:
    logger.error(f"Error loading NLP models: {e}")
    nlp = None
    ner_model = None
    generator = None
    sentiment_analyzer = None

def search_sanctions_by_name(name):
    if sanctions_df.empty or len(sanctions_df.columns) < 2:
        logger.warning("Sanctions DataFrame is empty or does not have enough columns. Returning empty matches.")
        return pd.DataFrame()
    try:
        matches = sanctions_df[sanctions_df.iloc[:, 1].str.lower() == name.lower()]
        return matches
    except Exception as e:
        logger.error(f"Error in search_sanctions_by_name: {e}")
        return pd.DataFrame()

# Function to parse text input, updated for the sample input.txt
def parse_text_to_transactions(text: str) -> List[Dict[str, str]]:
    try:
        if nlp is None:
            logger.warning("SpaCy model not loaded. Using basic parsing.")
        
        transactions = text.split('---')
        structured_data_list = []
        
        for transaction in transactions:
            transaction = transaction.strip()
            if not transaction:
                continue
            
            lines = transaction.split('\n')
            data = {}
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Handle section headers
                if line.lower() in ['sender:', 'receiver:', 'additional notes:']:
                    current_section = line.lower().rstrip(':')
                    if current_section not in data:
                        data[current_section] = {}
                    continue
                
                # Parse key-value pairs
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if current_section:
                        data[current_section][key] = value
                    else:
                        if key == "transaction id":
                            data["TransactionID"] = value
                        elif key == "amount":
                            data["Amount"] = value
                        elif key == "transaction type":
                            data["TransactionType"] = value
                        elif key == "reference":
                            data["Reference"] = value
            
            # Map required fields from parsed data
            data["PayerName"] = data.get("sender", {}).get("name", "").strip('"')
            data["ReceiverName"] = data.get("receiver", {}).get("name", "").strip('"')
            
            # Construct TransactionDetails from TransactionType and Reference
            transaction_type = data.get("TransactionType", "")
            reference = data.get("Reference", "")
            data["TransactionDetails"] = f"{transaction_type} - {reference}" if transaction_type and reference else transaction_type or reference
            
            # Extract ReceiverCountry from Receiver Address or Account
            receiver_address = data.get("receiver", {}).get("address", "")
            receiver_account = data.get("receiver", {}).get("account", "")
            if receiver_address:
                match = re.search(r',\s*([A-Za-z\s]+)$', receiver_address)
                data["ReceiverCountry"] = match.group(1).strip() if match else "Unknown"
            elif receiver_account:
                match = re.search(r'\(([^)]+)\)$', receiver_account)
                data["ReceiverCountry"] = match.group(1).strip().split(',')[-1].strip() if match else "Unknown"
            else:
                data["ReceiverCountry"] = "Unknown"
            
            # Clean and validate Amount early
            if "Amount" in data and data["Amount"]:
                amount_str = data["Amount"].replace("$", "").replace(",", "").replace("(USD)", "").strip()
                try:
                    float(amount_str)  # Test if it’s a valid number
                    data["Amount"] = amount_str
                except ValueError:
                    logger.error(f"Invalid Amount format in text parsing: {data['Amount']}")
                    data["Amount"] = "0.0"  # Default to 0.0 if invalid
            else:
                data["Amount"] = "0.0"  # Default if missing
            
            # Ensure all required fields are present, defaulting to empty string if missing
            required_fields = ["TransactionID", "PayerName", "ReceiverName", "TransactionDetails", "Amount", "ReceiverCountry"]
            for field in required_fields:
                if field not in data or not data[field]:
                    data[field] = "" if field != "Amount" else "0.0"
            
            structured_data_list.append(data)
        
        return structured_data_list
    except Exception as e:
        logger.error(f"Error parsing text input: {e}")
        return []

def extract_entities(text):
    if nlp is None:
        logger.warning("SpaCy model not loaded. Skipping entity extraction.")
        return []
    try:
        doc = nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]
    except Exception as e:
        logger.error(f"Error in extract_entities: {e}")
        return []

# Sentiment analysis functions from sentimentanalysis.py
def chunk_text(text, max_tokens=4096):
    if sentiment_analyzer is None:
        return [text]
    tokens = sentiment_analyzer.tokenizer.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return chunks

section_keywords = ["Controversy", "Criticism", "Scandal", "Issues", "Problems"]
def extract_section_content(text, section_title_keywords):
    sections = re.split(r'(?=\n==+.*?==+)', text)
    selected_sections = []
    for section in sections:
        for keyword in section_title_keywords:
            if keyword.lower() in section.lower():
                selected_sections.append(section.strip())
                break
    return selected_sections

def analyze_sentiment_of_sections(sections):
    if sentiment_analyzer is None:
        logger.warning("Sentiment analyzer not loaded. Assuming neutral sentiment.")
        return []
    
    negative_sections = []
    for section in sections:
        chunks = chunk_text(section)
        for chunk in chunks:
            text_chunk = sentiment_analyzer.tokenizer.decode(chunk, skip_special_tokens=True)
            sentiment = sentiment_analyzer(text_chunk)
            if sentiment[0]['label'] == 'LABEL_0':  # Negative sentiment
                negative_sections.append(text_chunk)
    return negative_sections

def analyze_entity_sentiment(entity_name):
    try:
        endpoint = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "titles": entity_name,
            "explaintext": True,
            "redirects": 1
        }
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()

        pages = data["query"]["pages"]
        page_content = next(iter(pages.values()))
        if "extract" not in page_content:
            return {"has_negative_sentiment": False, "details": "No Wikipedia article found"}

        sections = extract_section_content(page_content["extract"], section_keywords)
        if not sections:
            return {"has_negative_sentiment": False, "details": "No relevant sections found"}

        negative_sections = analyze_sentiment_of_sections(sections)
        return {
            "has_negative_sentiment": bool(negative_sections),
            "details": negative_sections if negative_sections else "No negative sentiment sections found"
        }
    except Exception as e:
        logger.error(f"Error in analyze_entity_sentiment for {entity_name}: {e}")
        return {"has_negative_sentiment": False, "details": f"Error: {str(e)}"}
def calculate_risk_score(payer_name, receiver_name, country, anomaly):
    try:
        base_score = 20

        # Check if payer is in sanctions list
        payer_sanctioned = not search_sanctions_by_name(payer_name).empty
        if payer_sanctioned:
            base_score += 20

        # Check if receiver is in sanctions list
        receiver_sanctioned = not search_sanctions_by_name(receiver_name).empty
        if receiver_sanctioned:
            base_score += 20

        # If both payer and receiver are sanctioned, increase risk significantly
        if payer_sanctioned and receiver_sanctioned:
            base_score += 40  # Additional penalty for both being sanctioned

        # Anomaly detection contribution
        if anomaly == -1:
            base_score += 20

        # High-risk country contribution
        high_risk_countries = ["Panama", "Brazil"]
        if country in high_risk_countries:
            base_score += 20

        return min(base_score, 100)
    except Exception as e:
        logger.error(f"Error in calculate_risk_score: {e}")
        return 20  # Default to low risk if calculation fails

def classify_risk(risk_score):
    try:
        if risk_score <= 40:
            return "Low"
        elif risk_score <= 70:
            return "Medium"
        return "High"
    except Exception as e:
        logger.error(f"Error in classify_risk: {e}")
        return "Low"  # Default to Low if classification fails

def extract_country(text):
    try:
        match = re.search(r',\s*([A-Za-z\s]+)$', text)
        if match:
            return match.group(1).strip()
        match = re.search(r'\s*\(([^)]+)\)$', text)
        if match:
            country_part = match.group(1).split(',')[-1].strip()
            return country_part if country_part else match.group(1).strip()
        match = re.search(r'\b[A-Z]{2}\b', text)
        if match:
            return match.group(0)
        return "Unknown"
    except Exception as e:
        logger.error(f"Error in extract_country: {e}")
        return "Unknown"

def parse_transaction(block):
    try:
        lines = block.strip().split('\n')
        transaction_data = {}
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.lower() in ['sender:', 'receiver:', 'additional notes:']:
                current_section = line.lower().rstrip(':')
                if current_section == 'additional notes':
                    transaction_data[current_section] = []
                else:
                    transaction_data[current_section] = {}
                continue
            if re.match(r'[^:]+:', line):
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                if key in ['amount', 'transaction type', 'reference', 'currency exchange', 'sender ip', 'date']:
                    current_section = None
                if current_section:
                    if current_section == 'additional notes':
                        transaction_data[current_section].append(f"{key}: {value}")
                    else:
                        transaction_data[current_section][key] = value
                else:
                    transaction_data[key] = value
            elif re.match(r'^\d+:\d+:\d+$', line) and 'date' in transaction_data:
                transaction_data['date'] += ' ' + line
            elif current_section == 'additional notes' and line:
                transaction_data[current_section].append(line.strip())
        
        return transaction_data
    except Exception as e:
        logger.error(f"Error in parse_transaction: {e}")
        return {}

def process_transactions(input_file, output_file):
    try:
        with open(input_file, 'r') as f:
            content = f.read()
        blocks = re.split(r'---\s*', content)
        rows = []
        for block in blocks:
            if not block.strip():
                continue
            transaction_data = parse_transaction(block)
            transaction_id = transaction_data.get('transaction id', '').strip()
            payer_name = transaction_data.get('sender', {}).get('name', '').strip('"')
            receiver_name = transaction_data.get('receiver', {}).get('name', '').strip('"')
            amount = transaction_data.get('amount', '').strip()
            transaction_type = transaction_data.get('transaction type', '').strip()
            reference = transaction_data.get('reference', '').strip('"')
            transaction_details = f"{transaction_type} - {reference}" if transaction_type and reference else ''
            receiver_address = transaction_data.get('receiver', {}).get('address', '')
            receiver_account = transaction_data.get('receiver', {}).get('account', '')
            receiver_country = extract_country(receiver_address) if receiver_address else extract_country(receiver_account)
            rows.append([transaction_id, payer_name, receiver_name, transaction_details, amount, receiver_country])
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(["Transaction ID", "Payer Name", "Receiver Name", "Transaction Details", "Amount", "Receiver Country"])
            writer.writerows(rows)
        logger.info(f"Conversion complete. Output written to {output_file}")
    except Exception as e:
        logger.error(f"Error processing transactions: {e}")

# FastAPI setup
app = FastAPI()

@app.post("/process_transaction")
async def process_transaction(data: Dict[str, str]):
    try:
        logger.info(f"Received data: {data}")
        
        file_type = data.get("file_type")
        content = data.get("content")
        
        if not file_type or not content:
            raise HTTPException(status_code=400, detail="Missing file_type or content")

        if "csv" in file_type.lower():
            df = pd.read_csv(StringIO(content), sep=';')
            df.columns = df.columns.str.strip()
            column_mapping = {
                "Transaction ID": "TransactionID",
                "Payer Name": "PayerName",
                "Receiver Name": "ReceiverName",
                "Transaction Details": "TransactionDetails",
                "Amount": "Amount",
                "Receiver Country": "ReceiverCountry"
            }
            df.rename(columns=column_mapping, inplace=True)
        elif "text" in file_type.lower():
            transactions = parse_text_to_transactions(content)
            if not transactions:
                raise HTTPException(status_code=400, detail="No valid transactions parsed from text")
            df = pd.DataFrame(transactions)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use CSV or TXT.")

        required_fields = ["TransactionID", "PayerName", "ReceiverName", "TransactionDetails", "Amount", "ReceiverCountry"]
        if not all(col in df.columns for col in required_fields):
            missing_fields = [col for col in required_fields if col not in df.columns]
            logger.error(f"Missing required fields: {missing_fields}")
            raise HTTPException(status_code=400, detail=f"Missing required fields: {missing_fields}")

        results = []
        for _, row in df.iterrows():
            transaction = row.to_dict()
            
            # Clean and parse Amount
            try:
                amount_str = str(transaction["Amount"]).replace("$", "").replace(",", "")
                amount_str = re.sub(r'\s*\([^)]+\)$|\s*USD$', '', amount_str).strip()
                transaction["Amount"] = float(amount_str)
            except (ValueError, KeyError) as e:
                logger.error(f"Error parsing Amount: {e}")
                return {"error": f"Invalid Amount format: {transaction.get('Amount', 'unknown')}"}

            # Extract entities
            entities = extract_entities(transaction["TransactionDetails"])
            logger.info(f"Extracted entities: {entities}")

            # Analyze sentiment for Payer and Receiver
            payer_sentiment = analyze_entity_sentiment(transaction["PayerName"])
            receiver_sentiment = analyze_entity_sentiment(transaction["ReceiverName"])

            # Classify entity types for payer and receiver
            sender_category = None
            receiver_category = None

            classify_result = ec.classify_entity(transaction["PayerName"])
            if classify_result is not None:
                sender_category = classify_result[0]

            classify_result = ec.classify_entity(transaction["ReceiverName"])
            if classify_result is not None:
                receiver_category = classify_result[0]

            # Fallback logic: Check for "org", "ltd", "inc", "corp", "corporation", "partners", "llc", "holdings" in the name if category is None
            if sender_category is None:
                payer_name_lower = transaction["PayerName"].lower()
                if any(keyword in payer_name_lower for keyword in ["org", "ltd", "inc", "corp", "corporation", "partners", "llc", "holdings"]):
                    sender_category = "CORPORATION"
                else:
                    sender_category = "PERSON"  # Default to PERSON if no keywords are found

            if receiver_category is None:
                receiver_name_lower = transaction["ReceiverName"].lower()
                if any(keyword in receiver_name_lower for keyword in ["org", "ltd", "inc", "corp", "corporation", "partners", "llc", "holdings"]) or "alas chiriguanas" in receiver_name_lower:
                    receiver_category = "CORPORATION"
                else:
                    receiver_category = "PERSON"  # Default to PERSON if no keywords are found

            # Anomaly detection
            try:
                anomaly_data = pd.DataFrame({"Amount": [transaction["Amount"]]})
                model = IsolationForest(contamination=0.1)
                anomaly = model.fit_predict(anomaly_data)[0]
            except Exception as e:
                logger.error(f"Error in anomaly detection: {e}")
                anomaly = 1

            # Calculate risk score
            risk_score = calculate_risk_score(
                transaction["PayerName"],
                transaction["ReceiverName"],
                transaction["ReceiverCountry"],
                anomaly
            )
            logger.info(f"Transaction risk score: {risk_score}")

            # Classify risk category
            risk_category = classify_risk(risk_score)
            logger.info(f"Transaction risk category: {risk_category}")

            # Build evidence for the transaction
            payer_sanction = search_sanctions_by_name(transaction["PayerName"])
            receiver_sanction = search_sanctions_by_name(transaction["ReceiverName"])
            evidence = [
                {"source": "Transaction", "data": f"Amount: {transaction['Amount']}, Country: {transaction['ReceiverCountry']}"}
            ]

            # Add external sources based on country or entity type
            if transaction["ReceiverCountry"] == "Panama":
                evidence.append({"source": "Panama Papers Database", "data": "Potential shell company activity"})
            if not payer_sanction.empty or not receiver_sanction.empty:
                evidence.append({"source": "Sanctions List", "data": "Entity match found"})
            if sender_category == "CORPORATION" or receiver_category == "CORPORATION":
                evidence.append({"source": "OpenCorporates", "data": "Company registration data"})
                evidence.append({"source": "Company Website", "data": "Public company information"})
            evidence.append({"source": "Wikipedia", "data": "Additional context on entity background"})

            # Add sentiment analysis to evidence
            if payer_sentiment["has_negative_sentiment"]:
                evidence.append({"source": "Wikipedia Sentiment Analysis (Payer)", "data": f"Negative sentiment detected: {payer_sentiment['details']}"})
            if receiver_sentiment["has_negative_sentiment"]:
                evidence.append({"source": "Wikipedia Sentiment Analysis (Receiver)", "data": f"Negative sentiment detected: {receiver_sentiment['details']}"})

            # Generate a detailed reason based on evidence
            reason_parts = []
            for evidence_item in evidence:
                source = evidence_item["source"]
                if source == "Sanctions List":
                    if not payer_sanction.empty and not receiver_sanction.empty:
                        reason_parts.append(f"both {transaction['PayerName']} and {transaction['ReceiverName']} are part of the sanctions list")
                    elif not payer_sanction.empty:
                        reason_parts.append(f"{transaction['PayerName']} is part of the sanctions list")
                    elif not receiver_sanction.empty:
                        reason_parts.append(f"{transaction['ReceiverName']} is part of the sanctions list")
                elif source == "Panama Papers Database":
                    reason_parts.append(f"the transaction involves {transaction['ReceiverCountry']}, which is flagged in the Panama Papers Database for potential shell company activity")
                elif source == "Wikipedia":
                    reason_parts.append("additional context was found on Wikipedia indicating potential risk factors")
                elif source == "Wikipedia Sentiment Analysis (Payer)":
                    reason_parts.append(f"negative sentiment was detected for {transaction['PayerName']} on Wikipedia")
                elif source == "Wikipedia Sentiment Analysis (Receiver)":
                    reason_parts.append(f"negative sentiment was detected for {transaction['ReceiverName']} on Wikipedia")
            if anomaly == -1:
                reason_parts.append(f"the transaction amount of {transaction['Amount']} was flagged as an anomaly")

            # Combine reason parts into a coherent sentence
            if reason_parts:
                reason = f"Transaction {transaction['TransactionID']} has a {risk_category} risk score because {', and '.join(reason_parts)}."
            else:
                reason = f"Transaction {transaction['TransactionID']} has a {risk_category} risk score based on the transaction details."

            # Calculate ConfidenceScore based on evidence sources and risk score
            confidence_score = 0.7  # Base confidence
            evidence_weights = {
                "Sanctions List": 0.15,  # High reliability
                "Panama Papers Database": 0.1,  # Moderate reliability
                "Wikipedia": 0.05,  # Lower reliability
                "Wikipedia Sentiment Analysis (Payer)": 0.05,
                "Wikipedia Sentiment Analysis (Receiver)": 0.05,
                "OpenCorporates": 0.1,
                "Company Website": 0.05,
                "Transaction": 0.0  # Baseline, no additional confidence
            }

            # Add confidence based on the number and quality of evidence sources
            for evidence_item in evidence:
                source = evidence_item["source"]
                confidence_score += evidence_weights.get(source, 0)

            # Adjust confidence based on risk score clarity
            if risk_score > 80 or risk_score < 20:
                confidence_score += 0.05

            # Ensure confidence_score is between 0 and 1
            confidence_score = min(max(confidence_score, 0.0), 1.0)

            # Build the result for the transaction
            result = {
                "TransactionID": transaction["TransactionID"],
                "ExtractedEntity": [transaction["PayerName"], transaction["ReceiverName"]],
                "EntityType": [sender_category, receiver_category],
                "RiskScore": float(risk_score / 100),  # Normalize to 0-1 scale
                "RiskCategory": risk_category,
                "SupportingEvidence": json.dumps(evidence),  # Serialize as JSON string
                "ConfidenceScore": confidence_score,
                "Reason": reason
            }

            results.append(result)

        logger.info(f"Transaction processing result: {results}")
        return results

    except Exception as e:
        logger.error(f"Error in process_transaction: {e}", exc_info=True)
        return {"error": f"Error processing transaction: {str(e)}"}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)