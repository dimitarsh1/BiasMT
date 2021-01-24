#!/bin/sh

SRC=$1

cat $SRC | sed -r 's/^\*([^ ]+ )/UNK\1/g' | sed -r 's/ \*([^ ]+ )/ UNK\1/g' | \
        sed -r 's/^@([^ ]+ )/ERR\1/g' | sed -r 's/ @([^ ]+ )/ ERR\1/g' | \
        sed -r 's/^#([^ ]+ )/ERR\1/g' | sed -r 's/ #([^ ]+ )/ ERR\1/g'

