@REM In each of the following command lines, after "python" and the script name, 
@REM the 1st argument can be "Raw" or "SNR", indicating the type of processed video.
@REM The 2nd argument is "nbin".
@REM The 3rd argument is "th_residual".
@REM The 4th argument is "th_pertmin".
@REM The 5th argument is "flexible_alpha".

@REM The following commands generate the unmixed traces for the main figures.
@REM The following commands process ABO video (Fig 2).
python TUnCaT_multi_ABO.py Raw 1 0 1 1
python TUnCaT_multi_ABO.py SNR 1 0 1 1

@REM The following commands process NAOMi video (Fig 4).
python TUnCaT_multi_NAOMi.py Raw 1 0 1 1
python TUnCaT_multi_NAOMi.py SNR 1 0 1 1

@REM The following commands process one-photon video (Fig 5).
python TUnCaT_multi_1p.py Raw 1 0 1 1
python TUnCaT_multi_1p.py SNR 1 0 1 1

@REM The following commands process ABO video using fixed alpha strategy (S3 Fig).
python TUnCaT_multi_ABO.py Raw 1 0 1 0
python TUnCaT_multi_ABO.py SNR 1 0 1 0

@REM The following commands process ABO video (Fig 2).
python TUnCaT_multi_ABO_SUNS.py Raw 1 0 1 1
python TUnCaT_multi_ABO_SUNS.py SNR 1 0 1 1

@REM The following commands run TUnCaT using temporal downsampling (S5 Fig).
python TUnCaT_multi_ABO.py Raw 2 0 1 1
python TUnCaT_multi_ABO.py Raw 4 0 1 1
python TUnCaT_multi_ABO.py Raw 8 0 1 1
python TUnCaT_multi_ABO.py Raw 16 0 1 1
python TUnCaT_multi_ABO.py Raw 32 0 1 1
python TUnCaT_multi_ABO.py Raw 64 0 1 1
python TUnCaT_multi_ABO.py Raw 100 0 1 1

python TUnCaT_multi_ABO.py SNR 2 0 1 1
python TUnCaT_multi_ABO.py SNR 4 0 1 1
python TUnCaT_multi_ABO.py SNR 8 0 1 1
python TUnCaT_multi_ABO.py SNR 16 0 1 1
python TUnCaT_multi_ABO.py SNR 32 0 1 1
python TUnCaT_multi_ABO.py SNR 64 0 1 1
python TUnCaT_multi_ABO.py SNR 100 0 1 1

python TUnCaT_multi_NAOMi.py Raw 2 0 1 1
python TUnCaT_multi_NAOMi.py Raw 4 0 1 1
python TUnCaT_multi_NAOMi.py Raw 8 0 1 1
python TUnCaT_multi_NAOMi.py Raw 16 0 1 1
python TUnCaT_multi_NAOMi.py Raw 32 0 1 1
python TUnCaT_multi_NAOMi.py Raw 64 0 1 1
python TUnCaT_multi_NAOMi.py Raw 100 0 1 1

python TUnCaT_multi_NAOMi.py SNR 2 0 1 1
python TUnCaT_multi_NAOMi.py SNR 4 0 1 1
python TUnCaT_multi_NAOMi.py SNR 8 0 1 1
python TUnCaT_multi_NAOMi.py SNR 16 0 1 1
python TUnCaT_multi_NAOMi.py SNR 32 0 1 1
python TUnCaT_multi_NAOMi.py SNR 64 0 1 1
python TUnCaT_multi_NAOMi.py SNR 100 0 1 1

python TUnCaT_multi_1p.py Raw 2 0 1 1
python TUnCaT_multi_1p.py Raw 4 0 1 1
python TUnCaT_multi_1p.py Raw 8 0 1 1
python TUnCaT_multi_1p.py Raw 16 0 1 1
python TUnCaT_multi_1p.py Raw 32 0 1 1
python TUnCaT_multi_1p.py Raw 64 0 1 1
python TUnCaT_multi_1p.py Raw 100 0 1 1

python TUnCaT_multi_1p.py SNR 2 0 1 1
python TUnCaT_multi_1p.py SNR 4 0 1 1
python TUnCaT_multi_1p.py SNR 8 0 1 1
python TUnCaT_multi_1p.py SNR 16 0 1 1
python TUnCaT_multi_1p.py SNR 32 0 1 1
python TUnCaT_multi_1p.py SNR 64 0 1 1
python TUnCaT_multi_1p.py SNR 100 0 1 1
