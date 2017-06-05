
# caffe-PersonReID

The code is based on Caffe for the problem of Person Re-Identification. 

## Some Sample Results 

Following Retreival Results were obtained using features extracted from our model.

### Market-1501

Query | Retreived Images |
------| ---------------  |
![](ReID/sampleResults/0004_c1s6_016996_00.jpg) | ![](ReID/sampleResults/0004_c5s3_066212_01.jpg) ![](ReID/sampleResults/0004_c3s3_065619_02.jpg) ![](ReID/sampleResults/0033_c1s6_014296_04.jpg) ![](ReID/sampleResults/1136_c5s3_073612_06.jpg) ![](ReID/sampleResults/1301_c5s3_031815_02.jpg) ![](ReID/sampleResults/0804_c3s2_096753_01.jpg)   ![](ReID/sampleResults/1270_c1s5_051866_02.jpg) ![](ReID/sampleResults/1440_c1s6_007541_02.jpg) |
![](ReID/sampleResults/0005_c1s1_001351_00.jpg) | ![](ReID/sampleResults/0005_c4s1_006951_03.jpg) ![](ReID/sampleResults/0005_c5s1_000401_03.jpg) ![](ReID/sampleResults/1382_c2s3_039207_01.jpg)  ![](ReID/sampleResults/0699_c2s2_051212_02.jpg) ![](ReID/sampleResults/1382_c5s3_054015_02.jpg) ![](ReID/sampleResults/0479_c3s1_125433_04.jpg) ![](ReID/sampleResults/1183_c3s3_006637_02.jpg) ![](ReID/sampleResults/0174_c2s1_053051_04.jpg) |
![](ReID/sampleResults/0016_c1s1_001351_00.jpg) | ![](ReID/sampleResults/0455_c5s1_116245_02.jpg)  ![](ReID/sampleResults/0016_c6s1_011551_01.jpg)  ![](ReID/sampleResults/0473_c5s1_123020_00.jpg) ![](ReID/sampleResults/0646_c5s2_033930_02.jpg) ![](ReID/sampleResults/1395_c5s3_048415_02.jpg)  ![](ReID/sampleResults/0302_c5s1_067748_03.jpg) ![](ReID/sampleResults/0388_c5s1_091548_01.jpg) ![](ReID/sampleResults/1077_c5s2_144199_02.jpg)   |

*Currently, this repo contains only baseline codes, it will soon be updated with the full version containing our model.

*Base Code for Triplet loss layer taken from [caffe-video_triplet](https://github.com/xiaolonw/caffe-video_triplet).
