#!/bin/sh

#
# Quelques tests pour vérifier les options/paramètres
#

TEST=$(basename $0 .sh)

TMP=/tmp/$TEST
LOG=$TEST.log
V=${VALGRIND}		# appeler avec la var. VALGRIND à "" ou "valgrind -q"

exec 2> $LOG
set -x

fail ()
{
    echo "==> Échec du test '$TEST' sur '$1'."
    echo "==> Log : '$LOG'."
    echo "==> Exit"
    exit 1
}

DN=/dev/null

# tests élémentaires sur les options

$V ./detect					&& fail "pas d'arg"

$V ./detect -i1 -l 1 cat $DN			|| fail "syntaxe -i1 -l 1"

$V ./detect -l1 -i 1 cat $DN			|| fail "syntaxe -l 1 -i1"

$V ./detect -i 0 cat $DN			&& fail "intervalle nul"

$V ./detect -i -1 cat $DN			&& fail "intervalle négatif"

$V ./detect -l -1 cat $DN			&& fail "limite négative"

$V ./detect -i1 -l1				&& fail "pas de cmd"

$V ./detect -c -i1 -l1 cat $DN > $DN		|| fail "syntaxe -c"

$V ./detect -t %S -c -i1 -l1 cat $DN > $DN	|| fail "syntaxe -t %S"

exit 0
