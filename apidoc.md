Skip to content
Navigation Menu
mistralai
client-python

Type / to search
Code
Issues
7
Pull requests
4
Actions
Projects
Security
Insights
Owner avatar
client-python
Public
mistralai/client-python
Go to file
t
Add file
Folders and files
Name		
Latest commit
github-actions[bot]speakeasybot
github-actions[bot]
and
speakeasybot
ci: regenerated with OpenAPI Doc , Speakeasy CLI 1.451.1 (#166)
329ce8c
 · 
last week
History
.github
Merge pull request #117 from jtpio/patch-1
4 months ago
.speakeasy
ci: regenerated with OpenAPI Doc , Speakeasy CLI 1.451.1 (#166)
last week
.vscode
Release Version 1.0.0 (#124)
4 months ago
docs
regent main sdk - new speakeasy version
last week
examples
chore: 🐝 Update SDK - Generate MISTRALAI MISTRALAI-SDK 1.2.0 (#153)
last month
packages
use to UPPER_CASE naming style (#164)
last week
scripts
chore: 🐝 Update SDK - Generate MISTRALAI MISTRALAI-SDK 1.2.0 (#153)
last month
src/mistralai
ci: regenerated with OpenAPI Doc , Speakeasy CLI 1.451.1 (#166)
last week
.genignore
Release Version 1.0.0 (#124)
4 months ago
.gitattributes
Release Version 1.0.0 (#124)
4 months ago
.gitignore
fix fc streaming (#127)
4 months ago
CONTRIBUTING.md
Release Version 1.0.0 (#124)
4 months ago
LICENSE
Release Version 1.0.0 (#124)
4 months ago
MIGRATION.md
[API-47] Images support and other updates (#144)
3 months ago
OLD-README.md
Release Version 1.0.0 (#124)
4 months ago
README.md
regent main sdk - new speakeasy version
last week
RELEASES.md
ci: regenerated with OpenAPI Doc , Speakeasy CLI 1.451.1 (#166)
last week
USAGE.md
regent main sdk - new speakeasy version
last week
poetry.lock
updating dependencies
last week
poetry.toml
Release Version 1.0.0 (#124)
4 months ago
py.typed
Release Version 1.0.0 (#124)
4 months ago
pylintrc
chore: 🐝 Update SDK - Generate MISTRALAI MISTRALAI-SDK 1.2.0 (#153)
last month
pyproject.toml
ci: regenerated with OpenAPI Doc , Speakeasy CLI 1.451.1 (#166)
last week
Repository files navigation
README
Apache-2.0 license
Mistral Python Client
Migration warning
This documentation is for Mistral AI SDK v1. You can find more details on how to migrate from v0 to v1 here

API Key Setup
Before you begin, you will need a Mistral AI API key.

Get your own Mistral API Key: https://docs.mistral.ai/#api-access
Set your Mistral API Key as an environment variable. You only need to do this once.
# set Mistral API Key (using zsh for example)
$ echo 'export MISTRAL_API_KEY=[your_key_here]' >> ~/.zshenv

# reload the environment (or just quit and open a new terminal)
$ source ~/.zshenv
Summary
Mistral AI API: Our Chat Completion and Embeddings APIs specification. Create your account on La Plateforme to get access and read the docs to learn how to use it.

Table of Contents
Mistral Python Client
Migration warning
API Key Setup
SDK Installation
SDK Example Usage
Providers' SDKs Example Usage
Available Resources and Operations
Server-sent event streaming
File uploads
Retries
Error Handling
Server Selection
Custom HTTP Client
Authentication
Debugging
IDE Support
Development
Contributions
SDK Installation
The SDK can be installed with either pip or poetry package managers.

PIP
PIP is the default package installer for Python, enabling easy installation and management of packages from PyPI via the command line.

pip install mistralai
Poetry
Poetry is a modern tool that simplifies dependency management and package publishing by using a single pyproject.toml file to handle project metadata and dependencies.

poetry add mistralai
SDK Example Usage
Create Chat Completions
This example shows how to create chat completions.

# Synchronous Example
from mistralai import Mistral
import os

with Mistral(
    api_key=os.getenv("MISTRAL_API_KEY", ""),
) as s:
    res = s.chat.complete(model="mistral-small-latest", messages=[
        {
            "content": "Who is the best French painter? Answer in one short sentence.",
            "role": "user",
        },
    ])

    if res is not None:
        # handle response
        pass

The same SDK client can also be used to make asychronous requests by importing asyncio.

# Asynchronous Example
import asyncio
from mistralai import Mistral
import os

async def main():
    async with Mistral(
        api_key=os.getenv("MISTRAL_API_KEY", ""),
    ) as s:
        res = await s.chat.complete_async(model="mistral-small-latest", messages=[
            {
                "content": "Who is the best French painter? Answer in one short sentence.",
                "role": "user",
            },
        ])

        if res is not None:
            # handle response
            pass

asyncio.run(main())
Upload a file
This example shows how to upload a file.

# Synchronous Example
from mistralai import Mistral
import os

with Mistral(
    api_key=os.getenv("MISTRAL_API_KEY", ""),
) as s:
    res = s.files.upload(file={
        "file_name": "example.file",
        "content": open("example.file", "rb"),
    })

    if res is not None:
        # handle response
        pass

The same SDK client can also be used to make asychronous requests by importing asyncio.

# Asynchronous Example
import asyncio
from mistralai import Mistral
import os

async def main():
    async with Mistral(
        api_key=os.getenv("MISTRAL_API_KEY", ""),
    ) as s:
        res = await s.files.upload_async(file={
            "file_name": "example.file",
            "content": open("example.file", "rb"),
        })

        if res is not None:
            # handle response
            pass

asyncio.run(main())
Create Agents Completions
This example shows how to create agents completions.

# Synchronous Example
from mistralai import Mistral
import os

with Mistral(
    api_key=os.getenv("MISTRAL_API_KEY", ""),
) as s:
    res = s.agents.complete(messages=[
        {
            "content": "Who is the best French painter? Answer in one short sentence.",
            "role": "user",
        },
    ], agent_id="<value>")

    if res is not None:
        # handle response
        pass

The same SDK client can also be used to make asychronous requests by importing asyncio.

# Asynchronous Example
import asyncio
from mistralai import Mistral
import os

async def main():
    async with Mistral(
        api_key=os.getenv("MISTRAL_API_KEY", ""),
    ) as s:
        res = await s.agents.complete_async(messages=[
            {
                "content": "Who is the best French painter? Answer in one short sentence.",
                "role": "user",
            },
        ], agent_id="<value>")

        if res is not None:
            # handle response
            pass

asyncio.run(main())
Create Embedding Request
This example shows how to create embedding request.

# Synchronous Example
from mistralai import Mistral
import os

with Mistral(
    api_key=os.getenv("MISTRAL_API_KEY", ""),
) as s:
    res = s.embeddings.create(inputs=[
        "Embed this sentence.",
        "As well as this one.",
    ], model="Wrangler")

    if res is not None:
        # handle response
        pass

The same SDK client can also be used to make asychronous requests by importing asyncio.

# Asynchronous Example
import asyncio
from mistralai import Mistral
import os

async def main():
    async with Mistral(
        api_key=os.getenv("MISTRAL_API_KEY", ""),
    ) as s:
        res = await s.embeddings.create_async(inputs=[
            "Embed this sentence.",
            "As well as this one.",
        ], model="Wrangler")

        if res is not None:
            # handle response
            pass

asyncio.run(main())
More examples
You can run the examples in the examples/ directory using poetry run or by entering the virtual environment using poetry shell.

Providers' SDKs Example Usage
Azure AI
Prerequisites

Before you begin, ensure you have AZUREAI_ENDPOINT and an AZURE_API_KEY. To obtain these, you will need to deploy Mistral on Azure AI. See instructions for deploying Mistral on Azure AI here.

Here's a basic example to get you started. You can also run the example in the examples directory.

import asyncio
import os

from mistralai_azure import MistralAzure

client = MistralAzure(
    azure_api_key=os.getenv("AZURE_API_KEY", ""),
    azure_endpoint=os.getenv("AZURE_ENDPOINT", "")
)

async def main() -> None:
    res = await client.chat.complete_async( 
        max_tokens= 100,
        temperature= 0.5,
        messages= [
            {
                "content": "Hello there!",
                "role": "user"
            }
        ]
    )
    print(res)

asyncio.run(main())
The documentation for the Azure SDK is available here.

Google Cloud
Prerequisites

Before you begin, you will need to create a Google Cloud project and enable the Mistral API. To do this, follow the instructions here.

To run this locally you will also need to ensure you are authenticated with Google Cloud. You can do this by running

gcloud auth application-default login
Step 1: Install

Install the extras dependencies specific to Google Cloud:

pip install mistralai[gcp]
Step 2: Example Usage

Here's a basic example to get you started.

import asyncio
from mistralai_gcp import MistralGoogleCloud

client = MistralGoogleCloud()


async def main() -> None:
    res = await client.chat.complete_async(
        model= "mistral-small-2402",
        messages= [
            {
                "content": "Hello there!",
                "role": "user"
            }
        ]
    )
    print(res)

asyncio.run(main())
The documentation for the GCP SDK is available here.

Available Resources and Operations
Available methods
agents
complete - Agents Completion
stream - Stream Agents completion
batch
batch.jobs
list - Get Batch Jobs
create - Create Batch Job
get - Get Batch Job
cancel - Cancel Batch Job
chat
complete - Chat Completion
stream - Stream chat completion
classifiers
moderate - Moderations
moderate_chat - Moderations Chat
embeddings
create - Embeddings
files
upload - Upload File
list - List Files
retrieve - Retrieve File
delete - Delete File
download - Download File
get_signed_url - Get Signed Url
fim
complete - Fim Completion
stream - Stream fim completion
fine_tuning
fine_tuning.jobs
list - Get Fine Tuning Jobs
create - Create Fine Tuning Job
get - Get Fine Tuning Job
cancel - Cancel Fine Tuning Job
start - Start Fine Tuning Job
models
list - List Models
retrieve - Retrieve Model
delete - Delete Model
update - Update Fine Tuned Model
archive - Archive Fine Tuned Model
unarchive - Unarchive Fine Tuned Model
Server-sent event streaming
Server-sent events are used to stream content from certain operations. These operations will expose the stream as Generator that can be consumed using a simple for loop. The loop will terminate when the server no longer has any events to send and closes the underlying connection.

The stream is also a Context Manager and can be used with the with statement and will close the underlying connection when the context is exited.

from mistralai import Mistral
import os

with Mistral(
    api_key=os.getenv("MISTRAL_API_KEY", ""),
) as s:
    res = s.chat.stream(model="mistral-small-latest", messages=[
        {
            "content": "Who is the best French painter? Answer in one short sentence.",
            "role": "user",
        },
    ])

    if res is not None:
        with res as event_stream:
            for event in event_stream:
                # handle event
                print(event, flush=True)
File uploads
Certain SDK methods accept file objects as part of a request body or multi-part request. It is possible and typically recommended to upload files as a stream rather than reading the entire contents into memory. This avoids excessive memory consumption and potentially crashing with out-of-memory errors when working with very large files. The following example demonstrates how to attach a file stream to a request.

Tip

For endpoints that handle file uploads bytes arrays can also be used. However, using streams is recommended for large files.

from mistralai import Mistral
import os

with Mistral(
    api_key=os.getenv("MISTRAL_API_KEY", ""),
) as s:
    res = s.files.upload(file={
        "file_name": "example.file",
        "content": open("example.file", "rb"),
    })

    if res is not None:
        # handle response
        pass
Retries
Some of the endpoints in this SDK support retries. If you use the SDK without any configuration, it will fall back to the default retry strategy provided by the API. However, the default retry strategy can be overridden on a per-operation basis, or across the entire SDK.

To change the default retry strategy for a single API call, simply provide a RetryConfig object to the call:

from mistral.utils import BackoffStrategy, RetryConfig
from mistralai import Mistral
import os

with Mistral(
    api_key=os.getenv("MISTRAL_API_KEY", ""),
) as s:
    res = s.models.list(,
        RetryConfig("backoff", BackoffStrategy(1, 50, 1.1, 100), False))

    if res is not None:
        # handle response
        pass
If you'd like to override the default retry strategy for all operations that support retries, you can use the retry_config optional parameter when initializing the SDK:

from mistral.utils import BackoffStrategy, RetryConfig
from mistralai import Mistral
import os

with Mistral(
    retry_config=RetryConfig("backoff", BackoffStrategy(1, 50, 1.1, 100), False),
    api_key=os.getenv("MISTRAL_API_KEY", ""),
) as s:
    res = s.models.list()

    if res is not None:
        # handle response
        pass
Error Handling
Handling errors in this SDK should largely match your expectations. All operations return a response object or raise an exception.

By default, an API error will raise a models.SDKError exception, which has the following properties:

Property	Type	Description
.status_code	int	The HTTP status code
.message	str	The error message
.raw_response	httpx.Response	The raw HTTP response
.body	str	The response content
When custom error responses are specified for an operation, the SDK may also raise their associated exceptions. You can refer to respective Errors tables in SDK docs for more details on possible exception types for each operation. For example, the list_async method may raise the following exceptions:

Error Type	Status Code	Content Type
models.HTTPValidationError	422	application/json
models.SDKError	4XX, 5XX	*/*
Example
from mistralai import Mistral, models
import os

with Mistral(
    api_key=os.getenv("MISTRAL_API_KEY", ""),
) as s:
    res = None
    try:
        res = s.models.list()

        if res is not None:
            # handle response
            pass

    except models.HTTPValidationError as e:
        # handle e.data: models.HTTPValidationErrorData
        raise(e)
    except models.SDKError as e:
        # handle exception
        raise(e)
Server Selection
Select Server by Name
You can override the default server globally by passing a server name to the server: str optional parameter when initializing the SDK client instance. The selected server will then be used as the default on the operations that use it. This table lists the names associated with the available servers:

Name	Server
eu	https://api.mistral.ai
Example
from mistralai import Mistral
import os

with Mistral(
    server="eu",
    api_key=os.getenv("MISTRAL_API_KEY", ""),
) as s:
    res = s.models.list()

    if res is not None:
        # handle response
        pass
Override Server URL Per-Client
The default server can also be overridden globally by passing a URL to the server_url: str optional parameter when initializing the SDK client instance. For example:

from mistralai import Mistral
import os

with Mistral(
    server_url="https://api.mistral.ai",
    api_key=os.getenv("MISTRAL_API_KEY", ""),
) as s:
    res = s.models.list()

    if res is not None:
        # handle response
        pass
Custom HTTP Client
The Python SDK makes API calls using the httpx HTTP library. In order to provide a convenient way to configure timeouts, cookies, proxies, custom headers, and other low-level configuration, you can initialize the SDK client with your own HTTP client instance. Depending on whether you are using the sync or async version of the SDK, you can pass an instance of HttpClient or AsyncHttpClient respectively, which are Protocol's ensuring that the client has the necessary methods to make API calls. This allows you to wrap the client with your own custom logic, such as adding custom headers, logging, or error handling, or you can just pass an instance of httpx.Client or httpx.AsyncClient directly.

For example, you could specify a header for every request that this sdk makes as follows:

from mistralai import Mistral
import httpx

http_client = httpx.Client(headers={"x-custom-header": "someValue"})
s = Mistral(client=http_client)
or you could wrap the client with your own custom logic:

from mistralai import Mistral
from mistralai.httpclient import AsyncHttpClient
import httpx

class CustomClient(AsyncHttpClient):
    client: AsyncHttpClient

    def __init__(self, client: AsyncHttpClient):
        self.client = client

    async def send(
        self,
        request: httpx.Request,
        *,
        stream: bool = False,
        auth: Union[
            httpx._types.AuthTypes, httpx._client.UseClientDefault, None
        ] = httpx.USE_CLIENT_DEFAULT,
        follow_redirects: Union[
            bool, httpx._client.UseClientDefault
        ] = httpx.USE_CLIENT_DEFAULT,
    ) -> httpx.Response:
        request.headers["Client-Level-Header"] = "added by client"

        return await self.client.send(
            request, stream=stream, auth=auth, follow_redirects=follow_redirects
        )

    def build_request(
        self,
        method: str,
        url: httpx._types.URLTypes,
        *,
        content: Optional[httpx._types.RequestContent] = None,
        data: Optional[httpx._types.RequestData] = None,
        files: Optional[httpx._types.RequestFiles] = None,
        json: Optional[Any] = None,
        params: Optional[httpx._types.QueryParamTypes] = None,
        headers: Optional[httpx._types.HeaderTypes] = None,
        cookies: Optional[httpx._types.CookieTypes] = None,
        timeout: Union[
            httpx._types.TimeoutTypes, httpx._client.UseClientDefault
        ] = httpx.USE_CLIENT_DEFAULT,
        extensions: Optional[httpx._types.RequestExtensions] = None,
    ) -> httpx.Request:
        return self.client.build_request(
            method,
            url,
            content=content,
            data=data,
            files=files,
            json=json,
            params=params,
            headers=headers,
            cookies=cookies,
            timeout=timeout,
            extensions=extensions,
        )

s = Mistral(async_client=CustomClient(httpx.AsyncClient()))
Authentication
Per-Client Security Schemes
This SDK supports the following security scheme globally:

Name	Type	Scheme	Environment Variable
api_key	http	HTTP Bearer	MISTRAL_API_KEY
To authenticate with the API the api_key parameter must be set when initializing the SDK client instance. For example:

from mistralai import Mistral
import os

with Mistral(
    api_key=os.getenv("MISTRAL_API_KEY", ""),
) as s:
    res = s.models.list()

    if res is not None:
        # handle response
        pass
Debugging
You can setup your SDK to emit debug logs for SDK requests and responses.

You can pass your own logger class directly into your SDK.

from mistralai import Mistral
import logging

logging.basicConfig(level=logging.DEBUG)
s = Mistral(debug_logger=logging.getLogger("mistralai"))
You can also enable a default debug logger by setting an environment variable MISTRAL_DEBUG to true.

IDE Support
PyCharm
Generally, the SDK will work well with most IDEs out of the box. However, when using PyCharm, you can enjoy much better integration with Pydantic by installing an additional plugin.

PyCharm Pydantic Plugin
Development
Contributions
While we value open-source contributions to this SDK, this library is generated programmatically. Any manual changes added to internal files will be overwritten on the next generation. We look forward to hearing your feedback. Feel free to open a PR or an issue with a proof of concept and we'll do our best to include it in a future release.

About
Python client library for Mistral AI platform

Resources
 Readme
License
 Apache-2.0 license
 Activity
 Custom properties
Stars
 492 stars
Watchers
 26 watching
Forks
 105 forks
Report repository
Releases 16
python - v1.2.5 - 2024-12-04 15:32:18
Latest
last week
+ 15 releases
Packages
No packages published
Used by 1.9k
@RayRaf
@Adefolajuwon
@Djok161
@allenlee-37
@TinChung41
@d-oit
@chsubhasis
@allenwillmakeit
+ 1,893
Contributors
30
@Bam4d
@GaspardBT
@speakeasybot
@thehunmonkgroup
@github-actions[bot]
@pandora-s-git
@fuegoio
@jean-malo
@aac228
@hrjn
@theophilegervet
@sophiamyang
@cool-RR
@patrickvonplaten
+ 16 contributors
Languages
Python
100.0%
Footer
© 2024 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Docs
Contact
Manage cookies
Do not share my personal information
