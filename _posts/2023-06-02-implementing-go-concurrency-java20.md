---
layout: single
classes: wide
title:  "Implementing Go Concurrency In Java 20"
categories:
  - workshop

tags:
  - java
  - go
  - concurrency
---

Go language and Java are both concurrent programming languages that can handle multiple tasks simultaneously. Go language has a feature called goroutines, which are lightweight threads that can run in parallel. Java is working on a similar feature called virtual threads, which is part of Project Loom. Virtual threads are also lightweight threads that can run on top of the existing Java threads.


Virtual threads are a long-awaited feature for Java developers. They allow multiple tasks to run concurrently without blocking the underlying OS threads. Java 19 introduced virtual threads as a preview feature, and Java 20 continues to support them in preview mode. Virtual threads are expected to improve the performance and scalability of Java applications.


This article demonstrates how to rewrite go concurrency code using **Java 20** [2], which is the latest release of Java SE Platform. The goal is not only to reproduce the same results, but also to write Java code that is as close as possible to the go equivalent. The go concurrency code is taken from the **Official Go Tour** [1], which is an interactive tutorial that introduces the basic concepts and features of the go programming language.

In the rest of the article, in each subsection, a go code is displayed together with equivalent Java code. For the whole source code please refer to GitHub repository [3]. 

This article tries to answer the question of whether Java is as capable as Go in terms of concurrency.

## Goroutines

```go
package main

import (
	"fmt"
	"time"
)

func say(s string) {
	for i := 0; i < 5; i++ {
		time.Sleep(100 * time.Millisecond)
		fmt.Println(s)
	}
}

func main() {
	go say("world")
	say("hello")
}
```

`go say("world")` runs the function in a new goroutine. `say("hellow")` runs the function in current goroutine. 


```java
package org.example;

public class Goroutines {
    private static void say(String s) {
        for (var i = 0; i < 5; i++) {
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                // ignore
            }
            System.out.println(s);
        }
    }

    public static void main(String[] args) {
        Thread.startVirtualThread(() -> say("world"));
        say("hello");
    }
}
```

Java equivalent is straightforward.  `Thread.startVirtualThread(() -> say("world"))` runs the fuction in a new virtual thread.


## Channels

```go
package main

import "fmt"

func sum(s []int, c chan int) {
	sum := 0
	for _, v := range s {
		sum += v
	}
	c <- sum // send sum to c
}

func main() {
	s := []int{7, 2, 8, -9, 4, 0}

	c := make(chan int)
	go sum(s[:len(s)/2], c)
	go sum(s[len(s)/2:], c)
	x, y := <-c, <-c // receive from c

	fmt.Println(x, y, x+y)
}
```

Each goroutine computes a partial sum and sends it to a channel. The main function receives the partial sums and adds them to get the final result.

```java
package org.example;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;

public class Channels {
    private static int sum(List<Integer> s) {
        int sum = 0;
        for (int v : s) {
            sum += v;
        }
        return sum;
    }

    public static void main(String[] args) throws ExecutionException, InterruptedException {
        var numbers = Arrays.asList(7, 2, 8, -9, 4, 0);

        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            var f1 = executor.submit(() -> sum(numbers.subList(0, numbers.size() / 2)));
            var f2 = executor.submit(() -> sum(numbers.subList(numbers.size() / 2, numbers.size())));
            int x = f1.get();
            int y = f2.get();

            System.out.printf("%d %d %d \n", x, y, x + y);
        }
    }
}
```

Java doesn't have similar feature to channels. But they often can be imitated using BlockingQueue objects. 

ArrayBlockingQueue is suitable for this porpuse.

Akka framework offers Actors[4] which can be considered advanced channels. But actors are not as succinct.

Note that executor is closed only after all work is completed in try block. Submitted tasks are completed using virtual threads. Results are combined in main thread.


## Buffered Channels

```go
package main

import "fmt"

func main() {
	ch := make(chan int, 2)
	ch <- 1
	ch <- 2
	fmt.Println(<-ch)
	fmt.Println(<-ch)
}
```

Channels have a capacity. Default capacity is 1. Channels block writes when full. Similarly channels block reads when empty.


```java
package org.example;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

public class BufferedChannels {
    public static void main(String[] args) throws InterruptedException {
        BlockingQueue<Integer> ch = new ArrayBlockingQueue<>(2);

        ch.put(1);
        ch.put(2);

        System.out.println(ch.take());
        System.out.println(ch.take());
    }
}
```

ArrayBlockingQueue object has similar behaviour. So it is trivial.


## Range and Close

```go
package main

import (
	"fmt"
)

func fibonacci(n int, c chan int) {
	x, y := 0, 1
	for i := 0; i < n; i++ {
		c <- x
		x, y = y, x+y
	}
	close(c)
}

func main() {
	c := make(chan int, 10)
	go fibonacci(cap(c), c)
	for i := range c {
		fmt.Println(i)
	}
}
```

Channels can be used in for loops. The loop iterates upon a new entry. The loop terminates when the cannel is closed. 


```java
package org.example;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

public class RangeAndClose {
    private static void fibonacci(int n, BlockingQueue<Object> c) {
        int x = 0;
        int y = 1;

        for (int i = 0; i < n; i++) {
            try {
                c.put(x);
            } catch (InterruptedException e) {
                //ignore
            }
            int xx = x;
            x = y;
            y = xx + y;
        }

        try {
            c.put("Done");
        } catch (InterruptedException e) {
            //ignore
        }
    }

    public static void main(String[] args) throws InterruptedException {
        BlockingQueue<Object> c = new ArrayBlockingQueue<>(10);
        Thread.startVirtualThread(() -> fibonacci(c.remainingCapacity(), c));

        while (c.take() instanceof Integer i) {
            System.out.println(i);
        }
    }
}
```

For loop needs to be replaced with a while loop. Pattern matching feature is used as a trick to terminate the while loop. Note that BlockingQueue accepts objects, instead of Integers.

## Select

```go
package main

import "fmt"

func fibonacci(c, quit chan int) {
	x, y := 0, 1
	for {
		select {
		case c <- x:
			x, y = y, x+y
		case <-quit:
			fmt.Println("quit")
			return
		}
	}
}

func main() {
	c := make(chan int)
	quit := make(chan int)
	go func() {
		for i := 0; i < 10; i++ {
			fmt.Println(<-c)
		}
		quit <- 0
	}()
	fibonacci(c, quit)
}
```

Select is useful while waiting for multiple channels.


```java
package org.example;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

public class Select {
    private static void fibonacci(BlockingQueue<Integer> c, BlockingQueue<Integer> quit, BlockingQueue<Integer> signal) {
        final AtomicBoolean run = new AtomicBoolean(true);
        final AtomicInteger x = new AtomicInteger(0);
        final AtomicInteger y = new AtomicInteger(1);

        final Runnable runFib = () -> {
            try {
                c.put(x.get());

                int xx = x.get();
                x.set(y.get());
                y.set(y.get() + xx);

            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        };

        final Runnable runQuit = () -> {
            run.set(false);
            System.out.println("quit");
        };

        select(signal, run, runFib, runQuit);
    }

    private static void select(BlockingQueue<Integer> signal, AtomicBoolean run, Runnable... tasks) {
        try (ExecutorService executorService = Executors.newVirtualThreadPerTaskExecutor()) {
            executorService.submit(() -> {
                while (run.get()) {
                    try {
                        switch (signal.take()) {
                            case 0 -> tasks[0].run();
                            case 1 -> tasks[1].run();
                            default -> throw new IllegalStateException("Unexpected value from signal");
                        }
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            });
        }
    }

    private static void doForSelect(BlockingQueue<Integer> signal, int caseId, Runnable task) {
        try {
            signal.put(caseId);

            task.run();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        BlockingQueue<Integer> c = new ArrayBlockingQueue<>(1);
        BlockingQueue<Integer> quit = new ArrayBlockingQueue<>(1);
        BlockingQueue<Integer> signal = new ArrayBlockingQueue<>(1);

        Thread.startVirtualThread(() -> {
            for (int i = 0; i < 10; i++) {
                doForSelect(signal, 0, () -> {
                    try {
                        System.out.println(c.take());
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                });
            }

            doForSelect(signal, 1, () -> {
                try {
                    quit.put(0);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            });
        });

        fibonacci(c, quit, signal);
    }
}
```

Java offers nothing similar to select in go. A naive approach would be using busy waits to imitate the select. However it would not efficient and fall short to replicate the select.

The proposed approach is to introduce an additional queue to signal queue updates. This way underlying virtual thread waits until one of the channels are ready for read or write.

The select construct is imiated in select function. It expects a queue for signalling, an atomic boolean for completion of the loop and an open ended list of tasks to run. 

The task to run is selected according to number read from the signal queue. Note that BlockingQueue operations are blocking thus it is necessary to run the right runnable.  


## Default Selection

```go
package main

import (
	"fmt"
	"time"
)

func main() {
	tick := time.Tick(100 * time.Millisecond)
	boom := time.After(500 * time.Millisecond)
	for {
		select {
		case <-tick:
			fmt.Println("tick.")
		case <-boom:
			fmt.Println("BOOM!")
			return
		default:
			fmt.Println("    .")
			time.Sleep(50 * time.Millisecond)
		}
	}
}
```

Default block is executed without any blocking. It is executed continuously when other cases are not applicable. 


```java
package org.example;

import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;

public class DefaultSelection {

    private static void doForSelect(BlockingQueue<Integer> signal, int caseId, Runnable task) {
        try {
            signal.put(caseId);

            task.run();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    private static void select(BlockingQueue<Integer> signal, AtomicBoolean run, Runnable defaultRunnable, Runnable... tasks) {
        try (ExecutorService executorService = Executors.newVirtualThreadPerTaskExecutor()) {
            executorService.submit(() -> {
                while (run.get()) {

                    switch (signal.poll()) {
                        case null -> defaultRunnable.run();
                        case 0 -> tasks[0].run();
                        case 1 -> tasks[1].run();
                        default -> throw new IllegalStateException("Unexpected value from signal");
                    }
                }
            });
        }
    }

    public static void main(String[] args) {
        final ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor();

        BlockingQueue<Integer> tick = new ArrayBlockingQueue<>(1);
        BlockingQueue<Integer> boom = new ArrayBlockingQueue<>(1);
        BlockingQueue<Integer> signal = new ArrayBlockingQueue<>(1);

        final var f0 = scheduler.scheduleAtFixedRate(() -> doForSelect(signal, 0, () -> {
            try {
                tick.put(0);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }), 100, 100, TimeUnit.MILLISECONDS);

        final var f1 = scheduler.schedule(() -> doForSelect(signal, 1, () -> {
            try {
                boom.put(0);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }), 500, TimeUnit.MILLISECONDS);


        AtomicBoolean run = new AtomicBoolean(true);

        Runnable defaultRunnable = () -> {
            System.out.println("    .");
            try {
                Thread.sleep(50);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        };

        Runnable runnTick = () -> {
            try {
                tick.take();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            System.out.println("tick.");
        };

        Runnable runnBoom = () -> {
            try {
                boom.take();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            System.out.println("BOOM!");
            run.set(false);
            f0.cancel(true);
            f1.cancel(true);
            scheduler.close();
        };

        select(signal, run, defaultRunnable, runnTick, runnBoom);
    }
}
```

Default case itroduces a diffent semantic. Previous implementation of select aimed to avoid busy waits. When default block is present, previous approach does not suffice.

It is not practical to make a blocking call to the signal queue. poll function is used as the non blocking alternative of take function. 

Note that java also does not jave time constructs similar to the ones used in the go example. SheduledExecutor is used to imitate them. 

Apart from the mentioned diffences the approach is similar to previous example.

## Mutex

```go
package main

import (
	"fmt"
	"sync"
	"time"
)

// SafeCounter is safe to use concurrently.
type SafeCounter struct {
	mu sync.Mutex
	v  map[string]int
}

// Inc increments the counter for the given key.
func (c *SafeCounter) Inc(key string) {
	c.mu.Lock()
	// Lock so only one goroutine at a time can access the map c.v.
	c.v[key]++
	c.mu.Unlock()
}

// Value returns the current value of the counter for the given key.
func (c *SafeCounter) Value(key string) int {
	c.mu.Lock()
	// Lock so only one goroutine at a time can access the map c.v.
	defer c.mu.Unlock()
	return c.v[key]
}

func main() {
	c := SafeCounter{v: make(map[string]int)}
	for i := 0; i < 1000; i++ {
		go c.Inc("somekey")
	}

	time.Sleep(time.Second)
	fmt.Println(c.Value("somekey"))
}
```
A mutex is used to allow mutually exclusive object access.


```java
package org.example;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.locks.ReentrantLock;



public class SyncMutex {
    public static void main(String[] args) throws InterruptedException {
        var c = new SafeCounter();
        for (int i = 0; i <1000; i++) {
            Thread.startVirtualThread(()->c.inc("somekey"));
        }

        Thread.sleep(1000);
        System.out.println(c.value("somekey"));
    }
}

class SafeCounter {
    private final ReentrantLock lock = new ReentrantLock();
    private final Map<String, Integer> v = new HashMap<>();

    public void inc(String key) {
        lock.lock();
        try {
            int value = Optional.ofNullable(v.get(key)).orElse(0);
            v.put(key, value + 1);
        } finally {
            lock.unlock();
        }
    }

    public int value(String key) {
        lock.lock();
        try {
            return Optional.ofNullable(v.get(key)).orElse(0);
        } finally {
            lock.unlock();
        }
    }
}
```

Java has many locking mechanisms available. Semaphores are one alternative. Reentrant lock is preferred here. 

## Exercises

Equivalent Binary Trees and Web Crawler exercies are not included. However they can be found in GitHub repository[3].


## Conclusion 

Is Java as capable as Go in terms of concurrency? With the addition of virtual threads, the answer is YES.

Go code is definetly more concise. Java developer needs to type more in order to achive similar results. The rich standard library may compensate for verbose nature of Java.

Apart from verbosity of Java, comparing number of lines in the examples may be misleading. Go has its own way of approaching concurrency. The same problem can be solved in many different ways. Trying to imitate the ways of go makes Java code look more lengthy. If the focus was to produce same results, java code would be shorter (compared to Java codes presented here).

### Personal Opinion

I have used java more than a decade to write many applications. I have used many languages but have not been as confident and as productive as I have been with Java. 

Java has many apis and many objects for concurrency which allows many approaches. However it makes the java api complicated. Doing concurrency right in Java is not as easy. Learning curve is not as flat. 

Take may word with a grain of salt. I am not confident in Go. However, its concurrency minded design is very appealing. The api is simple and concise.

In terms of concurrency Go seems to be more promising. 



## References
1. [Go Concurrency Tour](https://go.dev/tour/concurrency/1)
2. [Java 20 Api Documentation](https://docs.oracle.com/en/java/javase/20/docs/api/index.html)
3. [Source Code](https://github.com/habanoz/go-java20-concurrency/tree/main)
4. [Akka Actors](https://doc.akka.io/docs/akka/current/typed/actors.html#akka-actors)
