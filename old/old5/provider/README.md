Providers are the abstraction layer between every LLM. Given a URI such as `openai://openai/gpt-4-turbo`, a `Provider` is *instantiated* by the scheme (`openai`) and *parameterized* by the hostname (`openai`, representing the default) and port (`80` by default, in this case). The path represents the model to use by the provider. One provider represents one connection to an LLM provider; if, say, two models are being served on separate ports, that requires two providers. If one location serves multiple models (such as openai's API), only one provider is needed and multiple models are instantiated from that. Additional parameters may be provided by the query string, ala `openai:///gpt-4-turbo?T=0.7`.

Example URIs:
- `openai:///gpt-3.5-turbo-0613?top_p=0.9` - omitting the netloc uses openai by default.
- `openai://azure/gpt-3.5-turbo` - use an azure instance (planned by unsupported)
- `openai://localhost:8031/mixtral` - openai-compatible locally hosted API on port 8031 using the "mixtral" model

A `Model` object bundles a provider with configuration required to use that particular model.