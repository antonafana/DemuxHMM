vcf_in=$1
vcf_out_stem=$2

for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 X
do
echo Doing $i
bcftools view ${vcf_in} --regions ${i} -o ${vcf_out_stem}${i}.vcf.gz -Oz
tabix -p vcf ${vcf_out_stem}${i}.vcf.gz
done

