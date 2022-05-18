# Connecting to Azure

## Authentication

The `hi-ml` package uses two possible ways of authentication with Azure.
The default is what is called "Interactive Authentication". When you submit a job to Azure via `hi-ml`, this will
use the credentials you used in the browser when last logging into Azure. If there are no credentials yet, you should
see instructions printed out to the console about how to log in using your browser.

We recommend using Interactive Authentication.

Alternatively, you can use a so-called Service Principal, for example within build pipelines.


## Service Principal Authentication

A Service Principal is a form of generic identity or machine account. This is essential if you would like to submit
training runs from code, for example from within an Azure pipeline. You can find more information about application registrations and service principal objects
[here](https://docs.microsoft.com/en-us/azure/active-directory/develop/app-objects-and-service-principals).

If you would like to use Service Principal, you will need to create it in Azure first, and then store 3 pieces
of information in 3 environment variables — please see the instructions below. When all the 3 environment variables are in place,
your Azure submissions will automatically use the Service Principal to authenticate.


### Creating the Service Principal

 1. Navigate back to [aka.ms/portal](https://aka.ms/portal)
 1. Navigate to `App registrations` (use the top search bar to find it).
 1. Click on `+ New registration` on the top left of the page.
 1. Choose a name for your application e.g. `MyServicePrincipal` and click `Register`.
 1. Once it is created you will see your application in the list appearing under `App registrations`. This step might take
 a few minutes.
 1. Click on the resource to access its properties. In particular, you will need the application ID.
 You can find this ID in the `Overview` tab (accessible from the list on the left of the page).
 1. Create an environment variable called `HIML_SERVICE_PRINCIPAL_ID`, and set its value to the application ID you
 just saw.
 1. You need to create an application secret to access the resources managed by this service principal.
 On the pane on the left find `Certificates & Secrets`. Click on `+ New client secret` (bottom of the page), note down your token.
 Warning: this token will only appear once at the creation of the token, you will not be able to re-display it again later.
 1. Create an environment variable called `HIML_SERVICE_PRINCIPAL_PASSWORD`, and set its value to the token you just
 added.

### Providing permissions to the Service Principal
Now that your service principal is created, you need to give permission for it to access and manage your AzureML workspace.
To do so:
1. Go to your AzureML workspace. To find it you can type the name of your workspace in the search bar above.
1. On the `Overview` page, there is a link to the Resource Group that contains the workspace. Click on that.
1. When on the Resource Group, navigate to `Access control`. Then click on `+ Add` > `Add role assignment`. A pane will appear on the
 the right. Select `Role > Contributor`. In the `Select` field type the name
of your Service Principal and select it. Finish by clicking `Save` at the bottom of the pane.


### Azure Tenant ID
The last remaining piece is the Azure tenant ID, which also needs to be available in an environment variable. To get
that ID:
1. Log into Azure
1. Via the search bar, find "Azure Active Directory" and open it.
1. In the overview of that, you will see a field "Tenant ID"
1. Create an environment variable called `HIML_TENANT_ID`, and set that to the tenant ID you just saw.
