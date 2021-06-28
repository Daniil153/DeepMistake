# Lexical Semantic Change Detection (LSCD) for the Russian language by the DeepMistake team.
Lexical semantic change detection

This repository contains code to reproduce the best results from the paper:

Arefyev Nikolay, Maksim Fedoseev, Vitaly Protasov, Daniil Homskiy, Adis Davletov, Alexander Panchenko. ["DeepMistake: Which Senses are Hard to Distinguish for a Word­in­Context Model"](http://www.dialog-21.ru/media/5235/arefyevnplusetal133.pdf) in Computational Linguistics and Intellectual Technologies: Proceedings of the International Conference “Dialogue 2021”.

DeepMistake was 2nd best system in the [RuShiftEval-2021 competition](http://www.dialog-21.ru/media/5296/pivovarovalpluskutuzova151.pdf).

After the competition we improved the system and outperformed the winner of the competition (see the table below).

# Citation
If you use any part of the system, please, cite our paper above.

# Reproduction of the best results

## Installation
Clone repositories:
```shell script
git clone https://github.com/Daniil153/DeepMistake
cd DeepMistake
git clone https://github.com/davletov-aa/mcl-wic
```
Install requirements 
```shell script
pip install -r mcl-wic/requirements.txt
 ```
## The solution for the RuShiftEval-2021 shared task on LSCD.
Download [data](https://zenodo.org/record/4977798#.YMxeNCZRVH4). You can download data from the command line also:
```shell script
bash download_files.sh
```
Download models: 
```shell script
bash download_models.sh first_concat mean_dist_l1ndotn_MSE mean_dist_l1ndotn_CE
```
To reproduce the best result in evaluation you need use:
```shell script
bash eval_best_eval_model.sh
```
To reproduce the best result in post evaluation you need use:
```shell script
bash eval_best_post-eval_model.sh
```
To reproduce second the best result in post evaluation you need use:
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
            <th colspan=1><b>Script</b></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>first+concat on MCL<sup>en-acc</sup><sub>CE</sub> &rarr; RSS<sup>dev2-sentSpear</sup><sub>MSE</sub>, LinReg(https://zenodo.org/record/4981585/files/first_concat.zip)</td>
            <td>0.795</td>
            <td>0.812</td>
            <td>0.78</td>
            <td>0.795</td>
            <td>eval_best_eval_model.sh</td>
        </tr>
        <tr>
            <td>mean+dist_l1ndotn-hs0 on MCL<sup>nen-acc</sup><sub>CE</sub> &rarr; RSS<sup>dev2-sentSpear</sup><sub>MSE</sub>, Mean (https://zenodo.org/record/4992633/files/mean_dist_l1ndotn_MSE.zip)</td>
            <td>0.833</td>
            <td>0.839</td>
            <td>0.834</td>
            <td>0.826</td>
            <td>eval_2best_post-eval_model.sh</td>
        </tr>
        <tr>
            <td>mean+dist_l1ndotn-hs0 on MCL<sup>nen-acc</sup><sub>CE</sub> &rarr; RSS<sup>dev2-sentSpear</sup><sub>CE</sub>, Mean (https://zenodo.org/record/4992613/files/mean_dist_l1ndotn_CE.zip)</td>
            <td>0.85</td>
            <td>0.863</td>
            <td>0.854</td>
            <td>0.834</td>
            <td>eval_best_post-eval_model.sh</td>
        </tr>
    </tbody>
</table>


### Solution for SemEval 2020 Task1
In the process

## Train models
Also you can train the best three models with 
```shell script
train_best_eval_model.sh
train_best2_post-eval_model.sh
train_best_post-eval_model.sh
```
