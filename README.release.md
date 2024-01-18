# Spring 2024 ECE 408 Course Files

## Getting started with Delta system

We will be using NCSA's Delta system for all programming assignments in this course. Please refer to Delta's documentation at https://docs.ncsa.illinois.edu/systems/delta/en/latest/. Also, please refer to the instructions posted on course Canvas on how to obtain an account on Delta. Please obtain an account on Delta before proceeding with this repository. After your account is approved, `ssh yourusername@login.delta.ncsa.illinois.edu` to login into Delta. 

For those opting for connect with VSCode Remote - SSH extension, setup instructions to connect to DELTA are available at this link: https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/prog_env.html#remote-ssh.

  **Tips for VSCode: After entering your password for the first time, click on the blue (details) in the lower right corner to start Duo two-factor login.**
  
## Getting started with the course repository

These instructions imply that you have obtained an account on Delta and are attempting to work on the course materials on Delta's login node. In other words, this repository is to be cloned on Delta, not on your personal computer or some other lab workstation.

### Creating the GitHub repository

First, follow this link to establish a class repository: https://edu.cs.illinois.edu/create-gh-repo/sp24_ece408. This needs to be done only once. After this step is complete, you should be able to visit https://github.com/illinois-cs-coursework/sp24_ece408_NetID page (where NetID is your NetID) that shows the contents of your repository. Initially, this repository will be empty; it will be populated soon.

### Creating a PAT (Personal Access Token)

Users are required to either use a Personal Access Token (PAT) or SSH key for Github repositories. In order to generate a token, go to your Github settings at https://github.com/settings/tokens page and click "Generate new token" link. You will need to generate a classic PAT. Give it a meaningful name/note and set the expiration date to be 2-3 weeks past the end of the semester. Check the "repo" box under the "select scopes" menu and create the token. The token is the string that begins with ghp_ - copy it immediately as you will need it throughout the semester. If you accidentally leave the page without copying the token, delete the token and create a new one. Next, click the Configure SSO dropdown menu, and click the Authorize button next to illinois-cs-coursework to allow you to use your token on UIUC repositories.

### Cloning the Repository

In a Delta terminal, navigate to the location where you would like to keep your files for ECE 408, most likelly in your home directory. Run the following to clone your repository:

  `git clone git@github.com:illinois-cs-coursework/sp24_ece408_NETID ece408git` where NETID is your UIUC NetID.

You will be prompted to enter your username and password. Enter your Github username enter your Personal Access Token instead of your password.  This will clone the remote repository to "ece408git" on your computer where you will be able to retrieve/submit/work on assignments.

And finally add release repository so you can receive class assignments: 

  `cd ece408git`

  `git remote add release https://github.com/illinois-cs-coursework/sp24_ece408_.release.git`

You can run `git remote -v` to verify that the repository was added. 

Also, make sure to configure your repository. This is needed by the auto-grader to pull the correct versions of your submission, otherwise we may not know whose work we are grading: 

  `git config user.name "Your Name"`

  `git config user.email "NetID@illinois.edu"`

### Adding libWB library to your repository ###

Our lab assignments rely on a special library, called libWB, for testing your code. Therefore, you need to install this library into your repository folder so it can be found by the compiler when compiling your code.

First, go to your ece408git folder: 

  `cd; cd ece408git` 

Next, clone libWB library: 

  `git clone https://github.com/abduld/libwb.git`

And then compile it: 

  `cd libwb; make; cd ..`

### Retrieving Assignments ###

To retrieve (or update) released assignments, go to your ece408git folder and run the following:

  `git fetch release`

  `git merge release/main -m "some comment" --allow-unrelated-histories`

  `git push origin main`

where "some comment" is a comment for your submission. The last command pushes the newly merged files to your remote repository. If something ever happens to your repository and you need to go back in time, you will be able to revert your repository to when you first retrieved an assignment.
