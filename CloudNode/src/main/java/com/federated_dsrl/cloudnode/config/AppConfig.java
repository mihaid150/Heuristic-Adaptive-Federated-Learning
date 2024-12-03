package com.federated_dsrl.cloudnode.config;

import com.federated_dsrl.cloudnode.entity.CloudTraffic;
import com.federated_dsrl.cloudnode.tools.CoolingSchedule;
import org.apache.commons.lang3.time.StopWatch;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;


@Configuration
public class AppConfig {
    @Bean
    public StopWatch stopWatch() {return new StopWatch();}

    @Bean
    public CloudTraffic cloudTraffic() {
        return new CloudTraffic();
    }

    @Bean
    public CoolingSchedule coolingSchedule() {return new CoolingSchedule();}
}
