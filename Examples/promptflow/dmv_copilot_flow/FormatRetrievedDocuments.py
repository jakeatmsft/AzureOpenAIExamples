from promptflow import tool

@tool
def format_retrieved_documents(docs: object, maxTokens: int) -> str:
  formattedDocs = []
  strResult = ""
  for index, doc in enumerate(docs):
    formattedDocs.append({
      f"[doc{index}]": {
        "title": doc['title'],
        "content": doc['content']
      }
    })
    formattedResult = { "retrieved_documents": formattedDocs }
    nextStrResult = str(formattedResult)
    if (estimate_tokens(nextStrResult) > maxTokens):
      break
    strResult = nextStrResult
  
  return strResult

def estimate_tokens(text: str) -> int:
  return (len(text) + 2) / 3
