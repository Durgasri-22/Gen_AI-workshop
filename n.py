from groq import Groq
client = Groq()
print(client.models.list())
