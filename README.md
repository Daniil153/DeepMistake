# LexSubGen
Lexical semantic change detection

This repository contains the code to reproduce the results from the paper:

Arefyev Nikolay, Maksim Fedoseev, Vitaly Protasov, Daniil Homskiy, Adis Davletov, Alexander Panchenko. DeepMistake: Which Senses are Hard to Distinguish for a
Word­in­Context Model 



## Installation
Clone repository from github.com.
```shell script
git clone https://github.com/davletov-aa/mcl-wic
```

### Setup environment
1. Install requirements
    ```shell script
    pip install -r requirements.txt
    ```


## Results
Results of the lexical substitution task are presented in the following table. To reproduce them, follow the instructions above to install the correct dependencies. 


<table>
    <thead>
        <tr>
            <th rowspan=1><b>Model</b></th>
            <th colspan=1><b>RuShiftEval avg</b></th>
            <th colspan=1><b>RuShiftEval1</b></th>
            <th colspan=1><b>RuShiftEval2</b></th>
            <th colspan=1><b>RuShiftEval3</b></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>first+concat on RSS dev2−sentSpear, LinReg</td>
            <td>0.791</td>
            <td>0.798</td>
            <td>0.773</td>
            <td>0.803</td>
        </tr>
        <tr>
            <td>mean+dist_l1ndotn­hs0 on MCL nen−acc RSS, Mean</td>
            <td>0.823</td>
            <td>0.825</td>
            <td>0.821</td>
            <td>0.823</td>
        </tr>
        <tr>
            <td>mean+dist_l1ndotn­hs0 on MCL nen−acc RSS CE, Mean</td>
            <td>0.843</td>
            <td>0.846</td>
            <td>0.848</td>
            <td>0.836</td>
        </tr>
    </tbody>
</table>


### Results reproduction

## Word Sense Induction Results
<table>
    <thead>
        <tr>
            <th rowspan=2><b>Model</b></th>
            <th colspan=1><b>SemEval 2013</b></th>
            <th colspan=1><b>SemEval 2010</b></th>
        </tr>
        <tr>
            <th>AVG</th>
            <th>AVG</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>XLNet</td>
            <td>33.4</td>
            <td>52.1</td>
        </tr>
        <tr>
            <td>XLNet+embs</td>
            <td>37.3</td>
            <td>54.1</td>
        </tr>
    </tbody>
</table>

To reproduce these results use 2.3.0 version of transformers and the following command:
```shell script
bash scripts/wsi.sh
```

### Web application
You could use command line interface to run Web application.
```shell script
# Run main server
lexsubgen-app run --host HOST 
                  --port PORT 
                  [--model-configs CONFIGS] 
                  [--start-ids START-IDS] 
                  [--start-all] 
                  [--restore-session]
``` 
**Example:**
```shell script
# Run server and serve models BERT and XLNet. 
# For BERT create server for serving model and substitute generator instantly (load resources in memory).
# For XLNet create only server.
lexsubgen-app run --host '0.0.0.0' 
                  --port 5000 
                  --model-configs '["my_cool_configs/bert.jsonnet", "my_awesome_configs/xlnet.jsonnet"]' 
                  --start-ids '[0]'

# After shutting down server JSON file with session dumps in the '~/.cache/lexsubgen/app_session.json'.
# The content of this file looks like:
# [
#     'my_cool_configs/bert.jsonnet',
#     'my_awesome_configs/xlnet.jsonnet',
# ]
# You can restore it with flag 'restore-session'
lexsubgen-app run --host '0.0.0.0' 
                  --port 5000 
                  --restore-session
# BERT and XLNet restored now
```
##### Arguments:
|Argument           |Default|Description                                                                                   |
|-------------------|-------|----------------------------------------------------------------------------------------------|
|`--help`           |       |Show this help message and exit                                                               |
|`--host`           |       |IP address of running server host                                                             |
|`--port`           |`5000` |Port for starting the server                                                                  |
|`--model-configs`  |`[]`   |List of file paths to the model configs.                                                      |
|`--start-ids`      |`[]`   |Zero-based indices of served models for which substitute generators will be created           |
|`--start-all`      |`False`|Whether to create substitute generators for all served models                                 |
|`--restore-session`|`False`|Whether to restore session from previous Web application run                                  |


### FAQ
1. How to use gpu? - You can use environment variable CUDA_VISIBLE_DEVICES to use gpu for inference:
   ```export CUDA_VISIBLE_DEVICES='1'``` or ```CUDA_VISIBLE_DEVICES='1'``` before your command.
1. How to run tests? - You can use pytest: ```pytest tests```
