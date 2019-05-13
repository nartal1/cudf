# Java API for cudf

This project provides java bindings for cudf, to be able to process large amounts of data on 
a GPU.

## Dependency

We use classifiers for different versions of cuda.  The java API remains the same, the difference
is the native runtime it is compiled and linked against.


CUDA 9.2:
```xml
<dependency>
    <groupId>ai.rapids</groupId>
    <artifactId>cudf</artifactId>
    <version>0.8-SNAPSHOT</version>
</dependency>
```

CUDA 10.0:
```xml
<dependency>
    <groupId>ai.rapids</groupId>
    <artifactId>cudf</artifactId>
    <classifier>cuda10</classifier>
    <version>0.8-SNAPSHOT</version>
</dependency>
```

## Build From Source

Build the native code first, and make sure the a JDK is installed and available.

