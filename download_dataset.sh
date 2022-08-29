#!/bin/bash

#Download and setup gdrive
if [ ! -x "$gdrive"]; then
echo "Downloading gdrive for the linux amd64 system. If you use other systems, please select the correct one on https://github.com/prasmussen/gdrive/releases"
wget --no-check-certificate https://github.com/prasmussen/gdrive/releases/download/2.1.1/gdrive_2.1.1_linux_amd64.tar.gz
tar -xzf gdrive_2.1.1_linux_amd64.tar.gz
rm gdrive_2.1.1_linux_amd64.tar.gz
fi
./gdrive about

#Download dataset
Split="test_seen"
Tasks="all"
while getopts 's:p:t:h' OPT; do
    case $OPT in
        s) SaveFolder="$OPTARG";;
        p) Split="$OPTARG";;
        t) Tasks="$OPTARG";;
        h) 
        echo -e "OPTIONS:\n-s: The save path for dataset (required)"
        echo -e "-p: The split of dataset to download (optional, default: test_seen)"
        echo -e "-t: The tasks of dataset to download (optional, default: all)"
        exit 1;;
        ?) echo "Unrecognized arguments. Please use -h to check the useage.";;
    esac
done

if [ -z ${SaveFolder} ];then
        echo "Missing save path,exit"
        exit 1
fi

split_list=( "train" "valid_seen" "valid_unseen" "test_seen" "test_unseen" )
valid_input=false
for s in ${split_list[@]}
do 
    if [ $Split = $s ]; then
        valid_input=true
    fi
done

if [ ${valid_input} == false ]; then
    echo "Wrong split input. Please select one from '${split_list[@]}'"
    exit 1
fi

tasks_list=( "all" "pick" "stack" "shape_sorter" "drop" "wipe" "pour" "door" "drawer" )
valid_input=false
for t in ${tasks_list[@]}
do 
    if [ $Tasks = $t ]; then
        valid_input=true
    fi
done

if [ ${valid_input} == false ]; then
    echo "Wrong tasks input. Please select one from '${tasks_list[@]}'"
    exit 1
fi

echo "Save dataset into: $SaveFolder. The split is: $Split. The tasks are: $Tasks"

case "$Split" in
    "train")
    id="1RTzJZWO3TUtA2iH9bucPDUVEvjF2AL5l"
    SaveFolder="$SaveFolder/train"
    ;;
    "valid_seen")
    id="1h7wtA0aTuVeDZQFRouDuiDlFnjeNN0qV"
    SaveFolder="$SaveFolder/valid/seen"
    ;;
    "valid_unseen")
    id="1kK_xgfwVWm7liJtai4SO75L7OpA1GexP"
    SaveFolder="$SaveFolder/valid/unseen"
    ;;
    "test_seen")
    id="1tuGIlRm_0xUh1WZFlYjJmcN0QX6sf6Nl"
    SaveFolder="$SaveFolder/test/seen"
    ;;
    "test_unseen")
    id="1esNV1--eWiYRvAozrXQHY8tZhVkyFrQf"
    SaveFolder="$SaveFolder/test/unseen"
    ;;
esac

query_string="'$id' in parents"
if [ $Tasks != "all" ]; then
    query_string="$query_string and name contains '$Tasks'"
fi
echo $query_string
./gdrive download query --skip --path $SaveFolder "$query_string"