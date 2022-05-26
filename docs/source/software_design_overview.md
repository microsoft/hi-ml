# Software Development Process

This document provides a high-level overview of the software development process that our team uses.
For detailed guidance, please refer to the [coding guidelines](coding_guidelines.md).

The design and development of the software in this repository is roughly separated into an **initiation**,
**prototyping**, and a **finalization** phase. The initiation phase can be skipped for minor changes, for example an
update to documentation.

## Version control

Software code versioning is done via GitHub. The code with the highest level of quality control is in the "main" branch.
Ongoing development happens in separate branches. Once development in the branch is finished, a Pull Request (PR)
process is started to integrate the branch into "main".

## Initiation

During the initiation phase, the following steps are carried out:

- Collection of a set of requirements
- Creating a suggested design for the change
- Review of the design with member of the core team

The deliverables of this phase are a detailed design of the proposed change in a GitHub Issue or a separate document.

## Prototyping

The engineering owner of the proposed change will create a branch of the current codebase. This branch is separate from
the released (main) branch to not affect any current functionality. In this branch, the engineer will carry out the
changes as proposed in the design document.

The engineer will also add additional software tests at different levels (unit tests, integration tests) as appropriate
for the design. These tests ensure that the proposed functionality will be maintained also in future.

The deliverable of this phase is a branch in the version control system that contains all proposed changes and a set of
software tests.

### Finalization

At this point, the engineering owner of the proposed change has carried out all necessary changes in a branch of the
codebase. They would now initiate a Pull Request process that consists of the following steps:

- The code will be reviewed by at least 2 engineers. Both need to provide their explicit approval before the proposed
  change can be integrated in the "main" branch.
- All unit and integration tests will start.
- All automatic code checks will start. These checks will verify

  - Consistency with static typing rules
  - Ensure that no passwords or other types of credentials are revealed
  - Ensure that none of the used third-party libraries contains high severity software vulnerabilities that could affect
    the proposed functionality.

For code to be accepted into the "main" branch, the following criteria need to be satisfied:

- All unit and integration tests pass.
- The code has been reviewed by at least 2 engineers who are members of the core development team.
- Any comments that have been added throughout the review process need to be resolved.
- All automated checks pass.

Once all the above criteria are satisfied, the branch will be merged into "main".

## Software Configuration Management

Software of Unknown Provenance (SOUP, 3rd party software that the main software code relies upon) is version controlled
as is the rest of the codebase. SOUP is consumed via two package management systems:

- Conda
- PyPi

Both of those package management systems maintain strict versioning: Once a version of a package is published, it
cannot be modified in place. Rather, a new version needs to be released.

Our training and deployment code uses Conda environment files that specify an explicit version of a dependent package to
use (for example, `lightning_bolts==0.4.0`)

dependencies to individual packages via
equality constraints, i.e., a
, unless coming from NuGet package repository. All nuget packages are version controlled as well and all SOUP config
files are in git too. These controls make all changes to SOUP go through the same control process and requirements as
changes to main code.

The list of SOUP items is maintained in Azure DevOps Components feature.

### Additional code quality control

Automatic code analysis tools (for example, StyleCop or FxCop) should run as part of the build process whenever possible. The configurations for those code analysis tools shall be set such that they produce build failures if a severe issue is found. The configuration for code analysis and style will be saved in source control for versioning purposes.

The developers shall attempt to create unit tests for each code change if it complements existing software verification protocols. Depending on the software safety classification per IEC 62304+A1:2015, the unit tests may or may not be considered as the formal design outputs [required for B, C].

Defect handling
---------------

The handling of any bugs or defects discovered during development or verification procedures in the iteration phase is done via Azure Dev Ops solution.

The defects are triaged and assigned during weekly meetings to the appropriate team member.

Risk management
---------------

In addition, the model contraindications will be analyzing in the risk management activity.

- Which types of medical scans / patients are expected to not work at all with the present model?

- Which types of medical scans / patients are expected to negatively affect the model's accuracy? (for example, beam hardening artefacts for CT, implants, unusual organ location, tumour has consumed the whole organ)

- What sort of other imaging artefacts or anatomical variabilities are expected to affect the visual representation of the anatomy (e.g. implantables or specific patient preparation procedures)

For more details refer to the risk management plan.

Change control
--------------

Change control process is governed by SOP 4.2-4. This document provides further clarification.

All assets related to a model (technical file, software code) are maintained in the git repository. Thus, any change to any of these items require a pull request per the configuration management. The pull requests create an audit trail documenting the change in the project assets over the iteration/formalization phases.

These pull requests will be aggregated and added to the Change Control Request.

### Model performance

Changes that may affect model performance (such as changes to the algorithm or dataset) shall trigger the generation of comparison report (can be as simple as a single table) which compares the performance of new model to the performance of the past model, per metrics established according to the Validation procedure.

The engineering team takes on the responsibility to track the change in performance and notify business owners of any changes that are significant and may impact clinical performance.

### Regulatory impact

Any changes to the model may potentially trigger new regulatory filings. The team will use the any current regulatory
guidance to see if the changes require a new regulatory filing. For every change that does not require a regulatory
filing, we will log a note to file which will describe justification for making the change without new regulatory
process.
