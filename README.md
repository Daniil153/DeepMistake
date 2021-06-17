# DeepMistake at LSCD
Lexical semantic change detection

This repository contains the code to reproduce the results from the paper:

Arefyev Nikolay, Maksim Fedoseev, Vitaly Protasov, Daniil Homskiy, Adis Davletov, Alexander Panchenko. ["DeepMistake: Which Senses are Hard to Distinguish for a Word­in­Context Model"](http://www.dialog-21.ru/media/5235/arefyevnplusetal133.pdf),




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
            <td>first+concat on <span class="formula">MCL<sub>CE</sub><sup>en-acc</sup></span></span> &rarr; RSS dev2−sentSpear, LinReg</td>
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

