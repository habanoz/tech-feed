---
layout: single
title:  "Java and Go Rest Application Performance Comparison"
classes: wide
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
During test 10K virtual users are used for 30 seconds. Tests are repeated twice to avoid unlucky executions.

All code, confifuration files and commands can be found in Github repository [1].


## Docker Stats

| App              | Image Size(MB) | Startup Times(ms) | Initial Memory Consumption(MB) |
|------------------|----------------|-------------------|--------------------------------|
| go               | 14             | -                 | 4.5                            |
| micronaut        | 340            | 638               | 79.3                           |
| spring           | 345            | 1349              | 118                            |
| micronaut-native | 81             | 74                | 10                             |
| spring-native    | 99             | 31                | 28.4                           |


Native code does not require additional libraries/jars. Size of native images are smaller.

Native code starts much faster. This is not surprising. It is already compiled. It does not require starting a JVM and loading required jars. 

Native code consumes less momory. Initial memory consumption supports this claim. 

In all cases Go application seems to be better. Native java applications come next. 


## K6 Test Results
 

| App              | Executions | Interrupted | Avg(ms) | Min(ms) | Med(ms) | Max(ms) | p(90) (ms) | p(95) (ms) |
|------------------|------------|-------------|---------|---------|---------|---------|------------|------------|
| go               | 368416     | 0           | 798     | 240     | 746     | 2180    | 1180       | 1300       |
| micronaut        | 119519     | 0           | 2530    | 812     | 2450    | 5620    | 3430       | 3910       |
| spring           | 263404     | 1808        | 917     | 288     | 663     | 9400    | 1680       | 1890       |
| micronaut-native | 119004     | 0           | 2580    | 700     | 2550    | 4900    | 3380       | 3610       |
| spring-native    | 333064     | 1808        | 728     | 231     | 647     | 8310    | 979        | 1210       |


Load Test results are not as straightforward. 

Micronaut results are worst. Even native micronaut results are not improved.

Spring results are much better. Upon this, spring native results improves a lot. Actually spring native results are comparable to go results.

Spring results indicate a stability issue. Spring tests ended with some executions interrupted probably due not completing in time. Eventough spring test results are good, max values worst. 

## Conclusion

This particular test shows strengths of native code. Spring native in Graalvm performs as good as Go code. However, this single test is not enough to make any generalization.

Without pointing to any language we can make following claims:

Native code starts faster. Native code relies on smaller containers. Native applications start with less memory consumption. Native applications tend to perform better than their non-native forms. However, as in the demonstrated micronaut case, it is not always the case. 


## References
1. [Test Code](https://github.com/habanoz/java-go-rest-app-compare)
2. [Spring Native](https://docs.spring.io/spring-boot/docs/current/reference/html/native-image.html)
3. [Micronaut Graalvm Application](https://guides.micronaut.io/latest/micronaut-creating-first-graal-app-gradle-java.html)
4. [K6 Tool](https://k6.io/open-source/)