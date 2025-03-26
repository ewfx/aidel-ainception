from transformers import pipeline
import Data_Extraction as de
import json
from sentence_transformers import SentenceTransformer, util

def classify_entity(entity_name):

  qa_pipeline = pipeline("question-answering", model="deepset/bert-base-cased-squad2")
  embedding_model = SentenceTransformer("all-MiniLM-L6-v2") 

  category_keywords = {}
  with open('org_categories.json', 'r') as file:
    category_keywords = json.loads(file.read())


  value = de.get_entity_details(entity_name)
  if value is not None:
    (entity_id, response) = value
    title = de.get_wikipedia_title(entity_id, response)
    print(title)
    (title, content) = de.get_wikipedia_content(title)
    if content is not None:
      content = content[0:content.index('\n\n')]
      question = f"What industry does {entity_name} operate in?"
      answer = qa_pipeline(question=question, context=content)
      print(answer['answer'])
      answer_embedding = embedding_model.encode(str(answer))

      best_category = None
      best_similarity = -1

      for category, keywords in category_keywords.items():
          for keyword in keywords:
              keyword_embedding = embedding_model.encode(keyword)
              similarity = util.cos_sim(answer_embedding, keyword_embedding).item()
              
              if similarity > best_similarity:
                  best_similarity = similarity
                  best_category = category

      print(f"Predicted Category: {str(best_category).upper()} (Similarity: {best_similarity:.2f})")
        
      return (str(best_category).upper(), f"{best_similarity:.2f}")
  return None