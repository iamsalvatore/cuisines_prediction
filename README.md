## Predicting Cuisines of Recipes

Project page: [dme/2021/projects.html](https://www.inf.ed.ac.uk/teaching/courses/dme/2021/projects.html#predicting-cuisines-of-recipes)


## Setup
- Create new conda environment
  ```
  conda create -n dme python=3.8
  ```
- Activate conda envrionment
  ```
  conda activate dme
  ```
- Install required libraries and packages
  ```
  pip install -r requirements.txt
  ```

## Reviewing PRs
Quality reviews are really important. You should spend time reviewing the code your peers write (not just fixing their mistakes without saying anything). I've noticed we're lacking reviewing and everyone should be contributing to helping with the review burden.

If you can't review something because you don't understand what they're doing, there's something very wrong with their code... not you. Ask clarifying questions and suggest ways for them to make their code more interpretable. Request they put comments where comment are necessary.

If your code is being reviewed, don't be insulted or annoyed at requests to reformat/add comments. Air on the side of helping your colleagues understand your work.

**That being said**, don't be an arse about formatting/naming. We should all have [yapf](https://github.com/google/yapf) installed and be using the following settings, whatever that formats to is fine and style changes should rarely block merger.
- Install `yapf`
  ```
  pip install yapf
  ```
- PyCharm: add External Tools and File Watchers
  - Preference -> Tools -> External Tools -> Add
    -  Name: `yapf`
    -  Program: `<path to yapf>` (you can use command `which yapf` to find the path of `yapf`)
    -  Arguments: `-i $FilePath$`
    -  Advanced Options
       -  [x] Synchronize files after execution
  - Preference -> Tools -> File Watchers -> Add -> `<custom>`
    - Name: `yapf`
    - File type: `Python`
    - Program: `<path to yapf>` (you can use command `which yapf` to find the path of `yapf`)
    - Arguments: `-i $FilePath$`
    - Working directory: `$ProjectFileDir$`
    - Advanced Options
      - [x] Auto-save edited files to trigger the watcher
      - [x] Trigger the wathcer on external changes
      - Show console: `On error` 
- VSCode: add the follow to `settings.json` (see [documentation](https://code.visualstudio.com/docs/getstarted/settings#_settings-file-locations))
  ```
    "python.formatting.provider": "yapf",
    "editor.formatOnSave": true,
    "editor.tabSize": 2,
    "python.formatting.yapfArgs": [
        "--style",
        "{based_on_style: google, indent_width: 2, column_limit: 80}"
    ]
  ```
- Sublime Text: add the following
  ```
  {
    "on_save": true,
    "config": {
      "based_on_style": "google",
      "INDENT_WIDTH": 2,
      "COLUMN_LIMIT": 80,
    },
  }
  ```

- Command line
  ```
  yapf --style={"based_on_style":"google","INDENT_WIDTH":2,"COLUMN_LIMIT":80} -i <filename>
  ```