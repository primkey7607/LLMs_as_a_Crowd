# Products
#!/usr/bin/env bash

ERDIR=$(dirname "$0")
DATADIR=${ERDIR}/datasets
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

