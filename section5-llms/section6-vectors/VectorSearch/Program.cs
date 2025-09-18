using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using Microsoft.SemanticKernel.Connectors.InMemory;
using OpenAI;
using System.ClientModel;
using System.Reflection.Emit;
using VectorSearch;

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

// Create an embedding generator (text-embedding-3-small is an example)
IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator =
    new OpenAIClient(credential, options)
    .GetEmbeddingClient("openai/text-embedding-3-small")
    .AsIEmbeddingGenerator();

// Create and populate the vector store
var vectorStore = new InMemoryVectorStore();

var moviesStore = vectorStore.GetCollection<int, Movie>("movies");

await moviesStore.EnsureCollectionExistsAsync();

foreach (var movie in MovieData.Movies)
{
    // generate the embedding vector for the movie description
    movie.Vector = await embeddingGenerator.GenerateVectorAsync(movie.Description);

    // add the overall movie to the in-memory vector store's movie collection
    await moviesStore.UpsertAsync(movie);
}

//1-Embed the user’s query
//2-Vectorized search
//3-Returns the records

// generate the embedding vector for the user's prompt
var query = "I want to see family friendly movie";
//var query = "A science fiction movie about space travel";
var queryEmbedding = await embeddingGenerator.GenerateVectorAsync(query);

// search the knowledge store based on the user's prompt
var searchResults = moviesStore.SearchAsync(queryEmbedding, top: 2);

// see the results just so we know what they look like
await foreach (var result in searchResults)
{
    Console.WriteLine($"Title: {result.Record.Title}");
    Console.WriteLine($"Description: {result.Record.Description}");
    Console.WriteLine($"Score: {result.Score}");
    Console.WriteLine();
}

Console.ReadLine();