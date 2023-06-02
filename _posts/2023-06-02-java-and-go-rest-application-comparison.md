---
layout: single
title:  "Java and Go Rest Application Performance Comparison"
categories:
  - analysis

tags:
  - java
  - go
  - rest
  - native
  - k6
  - docker
  - graalvm 
---

Graalvm aims to improve the performance and efficiency of Java applications in the cloud. Graalvm enables Java applications to be compiled to native code, which is expected to run faster and use less memory than the traditional JVM. 

One might wonder how native Java code compares to Go code, which is also known for its speed and low memory footprint. This article tries to answer this question.

## Methodology

A simple rest application is developped using Go and Java languages. Java application comes with two variants one using Spring Boot 3 and other using Micronaut framework.

Additional to using traditional JVM, java applications are also tested in Graalvm after compiling to native code. 


Rest application has 3 methods, 2 of which used in the test. A post method is used to add a new record. A get method is used to return list of all records. 
Rest application does not involve any external system access to avoid external factors interfering with the test. All data is kept in memory. 

All applications are containerized using docker. Docker compose is used to start the containers. 

K6 tool is used for testing. Test script first makes a post call. The same object is used all the time. The post call is followed by a get call which fetches all available records.  

All code, confifuration files and commands can be found in Github repository [1].


## Docker Stats

| App              | Image Size(MB) | Startup Times(ms) | Initial Memory Consumption(MB) |
|------------------|----------------|-------------------|--------------------------------|
| go               | 14             | -                 | 4.5                            |
| micronaut        | 340            | 638               | 79.3                           |
| spring           | 345            | 1349              | 118                            |
| micronaut-native | 81             | 74                | 10                             |
| spring-native    | 99             | 31                | 28.4                           |



## K6 Test Results
 

| App              | Executions | Interrupted | Avg(ms) | Min(ms) | Med(ms) | Max(ms) | p(90) (ms) | p(95) (ms) |
|------------------|------------|-------------|---------|---------|---------|---------|------------|------------|
| go               | 368416     | 0           | 798     | 240     | 746     | 2180    | 1180       | 1300       |
| micronaut        | 119519     | 0           | 2530    | 812     | 2450    | 5620    | 3430       | 3910       |
| spring           | 263404     | 1808        | 917     | 288     | 663     | 9400    | 1680       | 1890       |
| micronaut-native | 119004     | 0           | 2580    | 700     | 2550    | 4900    | 3380       | 3610       |
| spring-native    | 333064     | 1808        | 728     | 231     | 647     | 8310    | 979        | 1210       |


## Conclusion

Concludes


## Outro

Readers are encouraged to check original reports to see real numbers.


## References
1. [Test Code](https://github.com/habanoz/java-go-rest-app-compare)
2. 