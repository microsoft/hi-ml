# Hello world javascript action

This action prints "Hello World" or "Hello" + the name of a person to greet to the log.

## Inputs

## `who-to-greet`

**Required** The name of the person to greet. Default `"World"`.

## Outputs

## `time`

The time we greeted you.

## Example usage

uses: actions/hello-world-javascript-action@v1.1
with:
  who-to-greet: 'Mona the Octocat'

## Build

Following the instructions: [https://docs.github.com/en/actions/creating-actions/creating-a-javascript-action](https://docs.github.com/en/actions/creating-actions/creating-a-javascript-action):

1. Install vercel/ncc by running this command in your terminal. npm i -g @vercel/ncc

1. Compile your index.js file. ncc build index.js --license licenses.txt

You'll see a new dist/index.js file with your code and the compiled modules. You will also see an accompanying dist/licenses.txt file containing all the licenses of the node_modules you are using.
