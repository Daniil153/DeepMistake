# DeepMistake at LSCD
Lexical semantic change detection

This repository contains the code to reproduce the results from the paper:

Arefyev Nikolay, Maksim Fedoseev, Vitaly Protasov, Daniil Homskiy, Adis Davletov, Alexander Panchenko. ["DeepMistake: Which Senses are Hard to Distinguish for a Word­in­Context Model"](http://www.dialog-21.ru/media/5235/arefyevnplusetal133.pdf),




## Installation
Clone repository from github.com.
```shell script
git clone https://github.com/Daniil153/DeepMistake
cd DeepMistake
git clone https://github.com/davletov-aa/mcl-wic
```

### Setup environment
1. Install requirements
    ```shell script
    pip install -r mcl-wic/requirements.txt
    ```
You can train the best three models with 
```shell script
train_best_eval_model.sh
train_best2_post-eval_model.sh
train_best_post-eval_model.sh
```

### Solution for RuShiftEval 2021
First, you need to download [word usages and other necessary files](https://zenodo.org/record/4977798#.YMxeNCZRVH4). Also you can use script:
```shell script
bash download_files.sh
```
For downloading weights of 3 best models:
```shell script
bash download_models.sh 
```
To reproduce the best result in evaluation you need use:
```shell script
bash eval_best_eval_model.sh
```
To reproduce the best result in post evaluation you need use:
```shell script
bash eval_best_post-eval_model.sh
```
To reproduce second the best result in post evaluation you need use
```shell script
bash eval_2best_post-eval_model.sh
```

## Results
Results of the LSCD task are presented in the following table. To reproduce them, follow the instructions above to install the correct dependencies. 


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
            <td>first+concat on MCL<sup>en-acc</sup><sub>CE</sub> &rarr; RSS<sup>dev2-sentSpear</sup><sub>MSE</sub>, LinReg(https://zenodo.org/record/4981585/files/first_concat.zip)</td>
            <td>0.791</td>
            <td>0.798</td>
            <td>0.773</td>
            <td>0.803</td>
        </tr>
        <tr>
            <td>mean+dist_l1ndotn-hs0 on MCL<sup>nen-acc</sup><sub>CE</sub> &rarr; RSS<sup>dev2-sentSpear</sup><sub>MSE</sub>, Mean (https://zenodo.org/record/4992633/files/mean_dist_l1ndotn_MSE.zip)</td>
            <td>0.823</td>
            <td>0.825</td>
            <td>0.821</td>
            <td>0.823</td>
        </tr>
        <tr>
            <td>mean+dist_l1ndotn-hs0 on MCL<sup>nen-acc</sup><sub>CE</sub> &rarr; RSS<sup>dev2-sentSpear</sup><sub>CE</sub>, Mean (https://zenodo.org/record/4992613/files/mean_dist_l1ndotn_CE.zip)</td>
            <td>0.843</td>
            <td>0.846</td>
            <td>0.848</td>
            <td>0.836</td>
        </tr>
    </tbody>
</table>

### Solution for SemEval 2020 Task1
