//  ------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
//  ------------------------------------------------------------------------------------------

const core = require('@actions/core');
const github = require('@actions/github');
const { promises: fs } = require('fs')

  async function run() {
  try {
    const sha = github.context.eventName == 'pull_request' ? github.context.payload.pull_request.head.sha : github.context.sha

    const coverage = await fs.readFile(core.getInput('file'), 'utf8')

    var table = ""
    const coverage_lines = coverage.split("\n")
    for (var i = 0; i < coverage_lines.length; i++)
    {
      const cells = coverage_lines[i].split(" ")
      var row = ""
      for (var j = 0; j < cells.length; j++)
      {
        if (cells[j] != "")
        {
          if (row.length != "")
          {
            row += "|"
          }
          row += cells[j]
        }
      }
      if (row != "")
      {
        table += row + "\n"
      }
    }

    console.log(`Table: ${table}`);

    const githubToken = core.getInput('token')
    if (githubToken === '') {
      return
    }

    const githubClient = github.getOctokit(githubToken)

    await githubClient.repos.createCommitComment({
      owner: github.context.repo.owner,
      repo: github.context.repo.repo,
      commit_sha: sha,
      body: table
    })
  } catch (error) {
    core.setFailed(error.message);
  }
}

run()
