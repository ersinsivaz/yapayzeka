using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using OpenAI;
using System.ClientModel;
using System.Numerics.Tensors;
using System.Text.Json.Serialization;

IConfigurationRoot configuration = new ConfigurationBuilder()
    .AddJsonFile("appsettings.json", optional: true, reloadOnChange: true)
    .AddUserSecrets<Program>()
    .Build();

var credential = new ApiKeyCredential(configuration["GithubModels:Token"]!);
var openAIUri = new Uri(configuration["OpenAI:Uri"]!);
var model = configuration["GithubModels:Model"];

var options = new OpenAIClientOptions()
{
    Endpoint = openAIUri
};

IChatClient chatClient = new OpenAIClient(credential, options)
    .GetChatClient(model)
    .AsIChatClient();

// Create an embedding generator (text-embedding-3-small is an example)
IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator =
    new OpenAIClient(credential, options)
    .GetEmbeddingClient("openai/text-embedding-3-small")
    .AsIEmbeddingGenerator();

//// 1: Generate a single embedding
//var embedding = await embeddingGenerator.GenerateVectorAsync("Hello, world!");
//Console.WriteLine($"Embedding dimensions: {embedding.Span.Length}");
//foreach (var value in embedding.Span)
//{
//    Console.Write("{0:0.00}, ", value);
//}

// Compare multiple embeddings using Cosine Similarity
var catVector = await embeddingGenerator.GenerateVectorAsync("cat");
var dogVector = await embeddingGenerator.GenerateVectorAsync("dog");
var kittenVector = await embeddingGenerator.GenerateVectorAsync("kitten");

Console.WriteLine($"cat-dog similarity: {TensorPrimitives.CosineSimilarity(catVector.Span, dogVector.Span):F2}");
Console.WriteLine($"cat-kitten similarity: {TensorPrimitives.CosineSimilarity(catVector.Span, kittenVector.Span):F2}");
Console.WriteLine($"dog-kitten similarity: {TensorPrimitives.CosineSimilarity(dogVector.Span, kittenVector.Span):F2}");

Console.ReadLine();
