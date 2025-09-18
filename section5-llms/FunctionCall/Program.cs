using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using OpenAI;
using System.ClientModel;

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

IChatClient chatClient = 
    new ChatClientBuilder(new OpenAIClient(credential,options).GetChatClient(model).AsIChatClient())
    .UseFunctionInvocation()
    .Build();

ChatOptions chatOptions = new ChatOptions { 

    Tools = [AIFunctionFactory.Create((string location,string unit) => 
    {
        var temperature = Random.Shared.Next(-20,40);
        var condition = new[] {"Sunny","Rainy","Cloudy","Windy","Snowy"}[Random.Shared.Next(0,4)];

        return $"The weather is {temperature} degrees C and {condition}";
    },
    "get_current_weather",
    "Get the current weather in a given location"
    )]
};

List<ChatMessage> chatHistory = [new(ChatRole.System, """
    You are a hiking enthusiast who helps people discover fun hikes in their area. 
    You are upbeat and friendly.
    """)];

// Weather conversation relevant to the registered function.
chatHistory.Add(new(ChatRole.User, """
    I live in Istanbul and I'm looking for a moderate intensity hike. 
    What's the current weather like? 
    """));

Console.WriteLine($"{chatHistory.Last().Role} >>> {chatHistory.Last()}");

ChatResponse response = await chatClient.GetResponseAsync(chatHistory, chatOptions);

chatHistory.Add(new(ChatRole.Assistant, response.Text));

Console.WriteLine($"{chatHistory.Last().Role} >>> {chatHistory.Last()}");
Console.ReadLine();