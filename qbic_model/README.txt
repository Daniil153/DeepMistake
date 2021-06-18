1. bash clone_git.sh
2. cd GramEval2020
3. bash env.sh 
4. source solution-env/bin/activate
5. cd ..
6. bash download.sh (загрузка лучшей модели)
	вроде после этого момента интернет не нужен
7. bash new_scr.sh --path <путь до файла для токенизации> <путь к файлу, который генерируется токенизатором, подробнее как выглядит - чуть ниже>



3 параметр в new_scr - название файла, который генерируется токенизатором
tokenize_<name_file>.conllu
т.е. для файла file.txt -> tokenize_file.conllu















