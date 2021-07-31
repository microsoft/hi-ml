# Examples

## Controlling when to submit to AzureML and when not

By default, the `hi-ml` package assumes that you supply a commandline argument `--azureml` (that can be anywhere on 
the commandline) to trigger a submission of the present script to AzureML. If you wish to control it via a different
flag, coming out of your own argument parser, use the `submit_to_azureml` argument of the function
`health.azure.himl.submit_to_azure_if_needed`. 

