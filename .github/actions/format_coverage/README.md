# Format coverage action

This action loads the file coverage.txt produced by pytest-cov, formats it as a markdown table, and adds it to the commit as a comment.

## Build

Following the instructions: [https://docs.github.com/en/actions/creating-actions/creating-a-javascript-action](https://docs.github.com/en/actions/creating-actions/creating-a-javascript-action):

1. Install nodejs and npm

1. Install vercel/ncc by running this command in your terminal. npm i -g @vercel/ncc

1. Compile your index.js file. ncc build index.js --license licenses.txt

You'll see a new dist/index.js file with your code and the compiled modules. You will also see an accompanying dist/licenses.txt file containing all the licenses of the node_modules you are using.
