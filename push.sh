#!/bin/bash
while getopts m: flag
do
    case "${flag}" in
        m) commit=${OPTARG};;
    esac
done

export https_proxy=http://127.0.0.1:2340;export http_proxy=http://127.0.0.1:2340;export all_proxy=socks5://127.0.0.1:2341
git add .
git commit -m "$commit"
git push -f origin gh-pages