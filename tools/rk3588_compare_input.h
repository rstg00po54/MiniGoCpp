#pragma once
// Shared exact input and five-head ONNX CPU reference for RK3588 CPU/RKNN tests.
// Built by prepare_rk3588_compare_input.py from the same network/input used by OpenCL.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
struct KataGoComparisonInput {
    std::vector<float> spatial,global,mask;
    std::vector<std::vector<float>> references;
    explicit KataGoComparisonInput(const std::string& filename) {
        std::ifstream f(filename,std::ios::binary);
        if(!f)throw std::runtime_error("Missing KataGo comparison input: "+filename);
        char magic[8]={};f.read(magic,8);
        if(!f||std::memcmp(magic,"K10IN001",8)!=0)
            throw std::runtime_error("Invalid KataGo comparison input format");
        auto read=[&](size_t count) {
            std::vector<float> x(count);
            f.read(reinterpret_cast<char*>(x.data()),count*sizeof(float));
            if(!f)throw std::runtime_error("Truncated shared comparison input");
            return x;
        };
        spatial=read(22*19*19);global=read(19);mask=read(19*19);
        for(size_t count:{size_t(1),size_t(361),size_t(3),size_t(4),size_t(361)})
            references.push_back(read(count));
        char extra;
        if(f.read(&extra,1))throw std::runtime_error("Extra bytes in shared comparison input");
    }
    static const char* head(size_t i) {
        static const char* names[]={"OutputPolicyPass","OutputPolicy","OutputValue","OutputScoreValue","OutputOwnership"};
        return names[i];
    }
    bool check(size_t index,const float* observed,size_t count,const char* backend)const {
        const auto& target=references.at(index);
        if(count!=target.size())throw std::runtime_error("Wrong KataGo NN output dimension");
        size_t wrong=0;double maxerr=0,mean=0;
        for(size_t j=0;j<count;j++) {
            double d=std::abs(double(observed[j])-double(target[j]));
            if(!std::isfinite(observed[j])||d>0.05+0.06*std::abs(target[j]))wrong++;
            maxerr=std::max(maxerr,d);mean+=d;
        }
        std::printf("COMPARE_VERIFY backend=%s output=%s wrong=%zu/%zu max_abs=%.6f mean_abs=%.6f\n",
            backend,head(index),wrong,count,maxerr,mean/count);
        return wrong==0;
    }
};
