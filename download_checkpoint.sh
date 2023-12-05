mkdir -p ./logs/finetune
FILENAME=./logs/finetune/pt_rd_80_ft_ub_mrv2.tar.gz
wget 'https://storage.worksmobile.com/k1/drive/r/24101/300038767/300038767/100001502685439/3472530196303545353?fileId=MTAwMDAxNTAyNjg1NDM5fDM0NzI1MzAxOTYzMDM1NDUzNTN8Rnw1MDAwMDAwMDAwMDg1MDg4MDAx&sharedLinkId=S-mv0CDPQ3--cMNA31T7aQ.p0XYy-DR71CTXrG_AK6HiN2k-ZpA-OI8muAcjUN6F1h1-FZG8sBIOhYXd74Vjxc_TA22EqAjwSw0_jkYxSbV_g&link=A' \
  -O $FILENAME
tar -zxvf $FILENAME -C ./logs/finetune/
rm -rf $FILENAME