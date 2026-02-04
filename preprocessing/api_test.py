from perplexity import Perplexity

client = Perplexity(api_key="api_key")

search = client.search.create(
    query=[
      "What is Comet Browser?",
      "Perplexity AI",
      "Perplexity Changelog"
    ]
)

for result in search.results:
    print(f"{result.title}: {result.url}")