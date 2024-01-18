# Spring 2024 ECE 408 Course Files

## Getting started with Delta system

We will be using NCSA's Delta system for all programming assignments in this course. Please refer to Delta's documentation at https://docs.ncsa.illinois.edu/systems/delta/en/latest/. Also, please refer to the instructions posted on course Canvas on how to obtain an account on Delta. Please obtain an account on Delta before proceeding with this repository. After your account is approved, `ssh yourusername@login.delta.ncsa.illinois.edu` to login into Delta. 

For those opting for connect with VSCode Remote - SSH extension, setup instructions to connect to DELTA are available at this link: https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/prog_env.html#remote-ssh.

  **Tips for VSCode: After entering your password for the first time, click on the blue details in the lower right corner to start Duo two-factor login.**

  <img width="340" alt="image" src="https://github.com/ECE-408-Course/ECE408SP24/assets/52022161/37ff36a3-e25f-4089-b6ae-4ded70daac72">
  
## Getting started with the course repository

These instructions imply that you have obtained an account on Delta and are attempting to work on the course materials on Delta's login node. In other words, this repository is to be cloned on Delta, not on your personal computer or some other lab workstation.

First, follow this link to establish a class repository: https://edu.cs.illinois.edu/create-gh-repo/sp24_ece408. This needs to be done only once. 

Second, clone your repository: `git clone git@github.com:illinois-cs-coursework/sp24_ece408_NETID ece408git` where NETID is your UIUC NetID.

Next, `cd ece408git` and clone libWB library: `git clone https://github.com/abduld/libwb.git`. 

Compile it: `cd libwb; make; cd ..`

And finally add release repository so you can receive class assignments: `git remote add release https://github.com/illinois-cs-coursework/sp24_ece408_.release.git`
