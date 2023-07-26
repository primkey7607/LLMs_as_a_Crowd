# Products
#!/usr/bin/env bash

ROOTDIR=$(dirname "$0")/..
ERDIR=${ROOTDIR}/llm_crowd/entity_resolution
DATADIR=${ROOTDIR}/data/entity_resolution
mkdir -p ${DATADIR}/train
mkdir -p ${DATADIR}/test

# WDC Products
curl -L https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/50pair.zip -o ${DATADIR}/products.zip
unzip ${DATADIR}/products.zip -d ${DATADIR}
for size in small medium large
do
    python ${ERDIR}/prep_products.py ${DATADIR}/wdcproducts50cc50rnd000un_train_${size}.json.gz ${DATADIR}/train/products-${size}.csv
    ln -s ./products-${size}.csv ${DATADIR}/train/products_seen-${size}.csv
    ln -s ./products-${size}.csv ${DATADIR}/train/products_half-${size}.csv
done
python ${ERDIR}/prep_products.py ${DATADIR}/wdcproducts50cc50rnd100un_gs.json.gz ${DATADIR}/test/products.csv
python ${ERDIR}/prep_products.py ${DATADIR}/wdcproducts50cc50rnd000un_gs.json.gz ${DATADIR}/test/products_seen.csv
python ${ERDIR}/prep_products.py ${DATADIR}/wdcproducts50cc50rnd050un_gs.json.gz ${DATADIR}/test/products_half.csv
rm ${DATADIR}/products.zip ${DATADIR}/wdcproducts50cc50rnd*.json.gz

# WDC LSPM
# We use the version checked into the ditto repo
DITTODIR=${ROOTDIR}/submodules/ditto
LSPMSOURCEDIR=${DITTODIR}/data/wdc
git submodule update --init ${DITTODIR}

for dataset in cameras computers shoes watches
do
    for size in small medium large xlarge
    do
        python ${ERDIR}/prep_lspm.py ${LSPMSOURCEDIR}/${dataset}/train.txt.${size} ${DATADIR}/train/${dataset}-${size}.csv
    done
    python ${ERDIR}/prep_lspm.py ${LSPMSOURCEDIR}/${dataset}/test.txt ${DATADIR}/test/${dataset}.csv
done
