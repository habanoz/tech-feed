---
layout: single
classes: wide
title:  "Evolution of Java: A case study"
categories:
  - workshop

tags:
  - java
---

Java was designed in the early 1990s with a specific set of goals and principles. Since then, the world of computing has changed dramatically, with new paradigms, technologies and challenges emerging. None of those developments could have been foreseen by the designers of Java. So, after so many years, has Java become obsolete? This is a question that many programmers and software engineers ask themselves, especially when they compare Java with newer and more expressive languages. 

In this article, we will try to demonstrate some aspects of how Java evolved over time to meet developer needs.


## Java 1.4

Following code uses Java 1.4 to read multiple text files concurrently and generate a map of word frequencies. At the end top 10 most frequent words are printed.

```java
package org.example;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;

public class ParallelWordCount implements Runnable {
    private static final int TOP_RESULTS = 10;
    private final String filename;
    private final Map wordCount;

    public ParallelWordCount(String filename) {
        this.filename = filename;
        this.wordCount = new HashMap();
    }

    public void run() {
        try {
            BufferedReader reader = new BufferedReader(new FileReader(filename));
            String line;
            while ((line = reader.readLine()) != null) {
                String[] words = line.split("\\s+");
                for (int i = 0; i < words.length; i++) {

                    String word = words[i];
                    if (word.length() == 0)
                        continue;

                    if (wordCount.containsKey(word)) {
                        wordCount.put(word, new Integer(((Integer) wordCount.get(word)).intValue() + 1));
                    } else {
                        wordCount.put(word, new Integer(1));
                    }
                }
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] fileNames) throws InterruptedException {
        int nFiles = fileNames.length;

        Thread[] threads = new Thread[nFiles];
        ParallelWordCount[] wordCounts = new ParallelWordCount[nFiles];

        for (int i = 0; i < nFiles; i++) {
            ParallelWordCount wc = new ParallelWordCount(fileNames[i]);
            wordCounts[i] = wc;
            threads[i] = new Thread(wc);
            threads[i].start();
        }
        for (int i = 0; i < nFiles; i++) {
            threads[i].join();
        }

        final Map totalWordCount = new HashMap();
        for (int i = 0; i < wordCounts.length; i++) {
            ParallelWordCount wc = wordCounts[i];
            for (Iterator iterator = wc.wordCount.entrySet().iterator(); iterator.hasNext(); ) {
                Map.Entry entry = (Map.Entry) iterator.next();
                String word = (String) entry.getKey();
                int count = ((Integer) entry.getValue()).intValue();
                if (totalWordCount.containsKey(word)) {
                    totalWordCount.put(word, new Integer(((Integer) totalWordCount.get(word)).intValue() + count));
                } else {
                    totalWordCount.put(word, new Integer(count));
                }
            }
        }


        List sortedWords = new ArrayList(totalWordCount.keySet());
        Collections.sort(sortedWords, new Comparator() {
            public int compare(Object o1, Object o2) {
                int count1 = ((Integer) totalWordCount.get(o1)).intValue();
                int count2 = ((Integer) totalWordCount.get(o2)).intValue();
                return count2 - count1;
            }
        });

        System.out.println("Top words:");
        for (int i = 0; i < TOP_RESULTS && i < sortedWords.size(); i++) {
            String word = (String) sortedWords.get(i);
            int count = ((Integer) totalWordCount.get(word)).intValue();

            System.out.println(word + ": " + count);
        }
    }
}
```


We will go through this code as we mention new features added to newer Java versions. 

## Java 5: Generics

```diff
Subject: [PATCH] generics added.
---
Index: src/main/java/org/example/ParallelWordCount.java
IDEA additional info:
Subsystem: com.intellij.openapi.diff.impl.patch.CharsetEP
<+>UTF-8
===================================================================
diff --git a/src/main/java/org/example/ParallelWordCount.java b/src/main/java/org/example/ParallelWordCount.java
--- a/src/main/java/org/example/ParallelWordCount.java	(revision 6ad2a382cacff423f0f5eb88f0cd4ae5b6e0e49d)
+++ b/src/main/java/org/example/ParallelWordCount.java	(revision b6d910df4832c0021a19235fa398f4c5baa3950f)
@@ -7,11 +7,11 @@
 public class ParallelWordCount implements Runnable {
     private static final int TOP_RESULTS = 10;
     private final String filename;
-    private final Map wordCount;
+    private final Map<String, Integer> wordCount;
 
     public ParallelWordCount(String filename) {
         this.filename = filename;
-        this.wordCount = new HashMap();
+        this.wordCount = new HashMap<String, Integer>();
     }
 
     public void run() {
@@ -27,7 +27,7 @@
                         continue;
 
                     if (wordCount.containsKey(word)) {
-                        wordCount.put(word, new Integer(((Integer) wordCount.get(word)).intValue() + 1));
+                        wordCount.put(word, new Integer(wordCount.get(word).intValue() + 1));
                     } else {
                         wordCount.put(word, new Integer(1));
                     }
@@ -55,15 +55,15 @@
             threads[i].join();
         }
 
-        final Map totalWordCount = new HashMap();
+        final Map<String, Integer> totalWordCount = new HashMap<String, Integer>();
         for (int i = 0; i < wordCounts.length; i++) {
             ParallelWordCount wc = wordCounts[i];
-            for (Iterator iterator = wc.wordCount.entrySet().iterator(); iterator.hasNext(); ) {
-                Map.Entry entry = (Map.Entry) iterator.next();
-                String word = (String) entry.getKey();
-                int count = ((Integer) entry.getValue()).intValue();
+            for (Iterator<Map.Entry<String, Integer>> iterator = wc.wordCount.entrySet().iterator(); iterator.hasNext(); ) {
+                Map.Entry<String, Integer> entry = iterator.next();
+                String word = entry.getKey();
+                int count = entry.getValue().intValue();
                 if (totalWordCount.containsKey(word)) {
-                    totalWordCount.put(word, new Integer(((Integer) totalWordCount.get(word)).intValue() + count));
+                    totalWordCount.put(word, new Integer(totalWordCount.get(word).intValue() + count));
                 } else {
                     totalWordCount.put(word, new Integer(count));
                 }
@@ -71,19 +71,19 @@
         }
 
 
-        List sortedWords = new ArrayList(totalWordCount.keySet());
-        Collections.sort(sortedWords, new Comparator() {
-            public int compare(Object o1, Object o2) {
-                int count1 = ((Integer) totalWordCount.get(o1)).intValue();
-                int count2 = ((Integer) totalWordCount.get(o2)).intValue();
+        List<String> sortedWords = new ArrayList<String>(totalWordCount.keySet());
+        Collections.sort(sortedWords, new Comparator<String>() {
+            public int compare(String w1, String w2) {
+                int count1 = totalWordCount.get(w1).intValue();
+                int count2 = totalWordCount.get(w2).intValue();
                 return count2 - count1;
             }
         });
 
         System.out.println("Top words:");
         for (int i = 0; i < TOP_RESULTS && i < sortedWords.size(); i++) {
-            String word = (String) sortedWords.get(i);
-            int count = ((Integer) totalWordCount.get(word)).intValue();
+            String word = sortedWords.get(i);
+            int count = totalWordCount.get(word).intValue();
 
             System.out.println(word + ": " + count);
         }
```

## Java 5: Enhanced For Loop


```diff
Subject: [PATCH] enhanced for loops
---
Index: src/main/java/org/example/ParallelWordCount.java
IDEA additional info:
Subsystem: com.intellij.openapi.diff.impl.patch.CharsetEP
<+>UTF-8
===================================================================
diff --git a/src/main/java/org/example/ParallelWordCount.java b/src/main/java/org/example/ParallelWordCount.java
--- a/src/main/java/org/example/ParallelWordCount.java	(revision b6d910df4832c0021a19235fa398f4c5baa3950f)
+++ b/src/main/java/org/example/ParallelWordCount.java	(revision 78df02ff9db1fa4cc3f67742473d4c83cf7ce930)
@@ -20,9 +20,8 @@
             String line;
             while ((line = reader.readLine()) != null) {
                 String[] words = line.split("\\s+");
-                for (int i = 0; i < words.length; i++) {
+                for (String word : words) {
 
-                    String word = words[i];
                     if (word.length() == 0)
                         continue;
 
@@ -51,15 +50,13 @@
             threads[i] = new Thread(wc);
             threads[i].start();
         }
-        for (int i = 0; i < nFiles; i++) {
-            threads[i].join();
+        for (Thread thread : threads) {
+            thread.join();
         }
 
         final Map<String, Integer> totalWordCount = new HashMap<String, Integer>();
-        for (int i = 0; i < wordCounts.length; i++) {
-            ParallelWordCount wc = wordCounts[i];
-            for (Iterator<Map.Entry<String, Integer>> iterator = wc.wordCount.entrySet().iterator(); iterator.hasNext(); ) {
-                Map.Entry<String, Integer> entry = iterator.next();
+        for (ParallelWordCount wc : wordCounts) {
+            for (Map.Entry<String, Integer> entry : wc.wordCount.entrySet()) {
                 String word = entry.getKey();
                 int count = entry.getValue().intValue();
                 if (totalWordCount.containsKey(word)) {
@@ -81,8 +78,7 @@
         });
 
         System.out.println("Top words:");
-        for (int i = 0; i < TOP_RESULTS && i < sortedWords.size(); i++) {
-            String word = sortedWords.get(i);
+        for (String word : sortedWords.subList(0, TOP_RESULTS)) {
             int count = totalWordCount.get(word).intValue();
 
             System.out.println(word + ": " + count);

```

## Java 5: Auto-boxing and Auto-unboxing

```diff
Subject: [PATCH] auto-boxing and auto-unboxing
---
Index: src/main/java/org/example/ParallelWordCount.java
IDEA additional info:
Subsystem: com.intellij.openapi.diff.impl.patch.CharsetEP
<+>UTF-8
===================================================================
diff --git a/src/main/java/org/example/ParallelWordCount.java b/src/main/java/org/example/ParallelWordCount.java
--- a/src/main/java/org/example/ParallelWordCount.java	(revision 78df02ff9db1fa4cc3f67742473d4c83cf7ce930)
+++ b/src/main/java/org/example/ParallelWordCount.java	(revision 00fa2e04ad4454be538834c778d0e5d4947cf486)
@@ -26,9 +26,9 @@
                         continue;
 
                     if (wordCount.containsKey(word)) {
-                        wordCount.put(word, new Integer(wordCount.get(word).intValue() + 1));
+                        wordCount.put(word, wordCount.get(word) + 1);
                     } else {
-                        wordCount.put(word, new Integer(1));
+                        wordCount.put(word, 1);
                     }
                 }
             }
@@ -58,11 +58,11 @@
         for (ParallelWordCount wc : wordCounts) {
             for (Map.Entry<String, Integer> entry : wc.wordCount.entrySet()) {
                 String word = entry.getKey();
-                int count = entry.getValue().intValue();
+                int count = entry.getValue();
                 if (totalWordCount.containsKey(word)) {
-                    totalWordCount.put(word, new Integer(totalWordCount.get(word).intValue() + count));
+                    totalWordCount.put(word, totalWordCount.get(word) + count);
                 } else {
-                    totalWordCount.put(word, new Integer(count));
+                    totalWordCount.put(word, count);
                 }
             }
         }
@@ -71,15 +71,15 @@
         List<String> sortedWords = new ArrayList<String>(totalWordCount.keySet());
         Collections.sort(sortedWords, new Comparator<String>() {
             public int compare(String w1, String w2) {
-                int count1 = totalWordCount.get(w1).intValue();
-                int count2 = totalWordCount.get(w2).intValue();
+                int count1 = totalWordCount.get(w1);
+                int count2 = totalWordCount.get(w2);
                 return count2 - count1;
             }
         });
 
         System.out.println("Top words:");
         for (String word : sortedWords.subList(0, TOP_RESULTS)) {
-            int count = totalWordCount.get(word).intValue();
+            int count = totalWordCount.get(word);
 
             System.out.println(word + ": " + count);
         }

```

## Java 1.5: java.util.concurrent package

The java.util.concurrent package, introduced in Java 5, provides very useful classes and utilities for parallel processing. This package enables developers to write concurrent programs that can leverage multiple cores and processors. Some of the key features of this package are thread pools, executors, synchronizers, concurrent collections, and atomic variables.

Atomic variables and ConcurrentHashMap are very suitable for our use case. They provide thread-safe and efficient operations on shared data without locking. They are also very useful for a wide range of use cases that require concurrency and scalability.

Note that putIfAbsent method is only present at  ConcurrentMap interface until it is added to Map interface at java 8.

```diff
Subject: [PATCH] concurrenthashmap and atomic integer
---
Index: src/main/java/org/example/ParallelWordCount.java
IDEA additional info:
Subsystem: com.intellij.openapi.diff.impl.patch.CharsetEP
<+>UTF-8
===================================================================
diff --git a/src/main/java/org/example/ParallelWordCount.java b/src/main/java/org/example/ParallelWordCount.java
--- a/src/main/java/org/example/ParallelWordCount.java	(revision 00fa2e04ad4454be538834c778d0e5d4947cf486)
+++ b/src/main/java/org/example/ParallelWordCount.java	(revision 9bfe9b71f85072dc536e7b5c2b6fddf82ac23c1d)
@@ -3,15 +3,17 @@
 import java.io.BufferedReader;
 import java.io.FileReader;
 import java.util.*;
+import java.util.concurrent.ConcurrentHashMap;
+import java.util.concurrent.atomic.AtomicInteger;
 
 public class ParallelWordCount implements Runnable {
     private static final int TOP_RESULTS = 10;
     private final String filename;
-    private final Map<String, Integer> wordCount;
+    private final Map<String, AtomicInteger> wordCount;
 
-    public ParallelWordCount(String filename) {
+    public ParallelWordCount(String filename, Map<String, AtomicInteger> wordCount) {
         this.filename = filename;
-        this.wordCount = new HashMap<String, Integer>();
+        this.wordCount = wordCount;
     }
 
     public void run() {
@@ -25,13 +27,11 @@
                     if (word.length() == 0)
                         continue;
 
-                    if (wordCount.containsKey(word)) {
-                        wordCount.put(word, wordCount.get(word) + 1);
-                    } else {
-                        wordCount.put(word, 1);
-                    }
+                    wordCount.putIfAbsent(word, new AtomicInteger());
+                    wordCount.get(word).getAndIncrement();
                 }
             }
+
             reader.close();
         } catch (Exception e) {
             e.printStackTrace();
@@ -43,9 +43,10 @@
 
         Thread[] threads = new Thread[nFiles];
         ParallelWordCount[] wordCounts = new ParallelWordCount[nFiles];
+        final Map<String, AtomicInteger> totalWordCount = new ConcurrentHashMap<String, AtomicInteger>();
 
         for (int i = 0; i < nFiles; i++) {
-            ParallelWordCount wc = new ParallelWordCount(fileNames[i]);
+            ParallelWordCount wc = new ParallelWordCount(fileNames[i], totalWordCount);
             wordCounts[i] = wc;
             threads[i] = new Thread(wc);
             threads[i].start();
@@ -54,32 +55,18 @@
             thread.join();
         }
 
-        final Map<String, Integer> totalWordCount = new HashMap<String, Integer>();
-        for (ParallelWordCount wc : wordCounts) {
-            for (Map.Entry<String, Integer> entry : wc.wordCount.entrySet()) {
-                String word = entry.getKey();
-                int count = entry.getValue();
-                if (totalWordCount.containsKey(word)) {
-                    totalWordCount.put(word, totalWordCount.get(word) + count);
-                } else {
-                    totalWordCount.put(word, count);
-                }
-            }
-        }
-
-
         List<String> sortedWords = new ArrayList<String>(totalWordCount.keySet());
         Collections.sort(sortedWords, new Comparator<String>() {
             public int compare(String w1, String w2) {
-                int count1 = totalWordCount.get(w1);
-                int count2 = totalWordCount.get(w2);
+                int count1 = totalWordCount.get(w1).intValue();
+                int count2 = totalWordCount.get(w2).intValue();
                 return count2 - count1;
             }
         });
 
         System.out.println("Top words:");
         for (String word : sortedWords.subList(0, TOP_RESULTS)) {
-            int count = totalWordCount.get(word);
+            int count = totalWordCount.get(word).get();
 
             System.out.println(word + ": " + count);
         }

```

Another major addition with java.util.concurrent package was ExecutorService API. Now we can get rid of manual thread management. 

```diff
Subject: [PATCH] use ExecutorService
---
Index: src/main/java/org/example/ParallelWordCount.java
IDEA additional info:
Subsystem: com.intellij.openapi.diff.impl.patch.CharsetEP
<+>UTF-8
===================================================================
diff --git a/src/main/java/org/example/ParallelWordCount.java b/src/main/java/org/example/ParallelWordCount.java
--- a/src/main/java/org/example/ParallelWordCount.java	(revision eb425f24f5410161a51b8399d8bde5851fc7ee11)
+++ b/src/main/java/org/example/ParallelWordCount.java	(revision 32f717e4145e733590ea18d64634c2a8ec53ac40)
@@ -4,6 +4,9 @@
 import java.io.FileReader;
 import java.util.*;
 import java.util.concurrent.ConcurrentHashMap;
+import java.util.concurrent.ExecutorService;
+import java.util.concurrent.Executors;
+import java.util.concurrent.TimeUnit;
 import java.util.concurrent.atomic.AtomicInteger;
 
 public class ParallelWordCount implements Runnable {
@@ -39,21 +42,15 @@
     }
 
     public static void main(String[] fileNames) throws InterruptedException {
-        int nFiles = fileNames.length;
-
-        Thread[] threads = new Thread[nFiles];
-        ParallelWordCount[] wordCounts = new ParallelWordCount[nFiles];
         final Map<String, AtomicInteger> totalWordCount = new ConcurrentHashMap<String, AtomicInteger>();
+        ExecutorService executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
 
-        for (int i = 0; i < nFiles; i++) {
-            ParallelWordCount wc = new ParallelWordCount(fileNames[i], totalWordCount);
-            wordCounts[i] = wc;
-            threads[i] = new Thread(wc);
-            threads[i].start();
-        }
-        for (Thread thread : threads) {
-            thread.join();
+        for (String fileName : fileNames) {
+            executorService.submit(new ParallelWordCount(fileName, totalWordCount));
         }
+
+        executorService.shutdown();
+        executorService.awaitTermination(60 * 5, TimeUnit.SECONDS);
 
         List<String> sortedWords = new ArrayList<String>(totalWordCount.keySet());
         Collections.sort(sortedWords, new Comparator<String>() {

```

## Java 1.5: Scanner


Another utility class Java 5 offers for our usecase is Scanner class which can make our file reading simpler.

```diff
Subject: [PATCH] use Scanner
---
Index: src/main/java/org/example/ParallelWordCount.java
IDEA additional info:
Subsystem: com.intellij.openapi.diff.impl.patch.CharsetEP
<+>UTF-8
===================================================================
diff --git a/src/main/java/org/example/ParallelWordCount.java b/src/main/java/org/example/ParallelWordCount.java
--- a/src/main/java/org/example/ParallelWordCount.java	(revision 32f717e4145e733590ea18d64634c2a8ec53ac40)
+++ b/src/main/java/org/example/ParallelWordCount.java	(revision 722034dbd9b2baf219e27eb5dfe9ab880a50d7b7)
@@ -1,7 +1,6 @@
 package org.example;
 
-import java.io.BufferedReader;
-import java.io.FileReader;
+import java.io.File;
 import java.util.*;
 import java.util.concurrent.ConcurrentHashMap;
 import java.util.concurrent.ExecutorService;
@@ -21,9 +20,9 @@
 
     public void run() {
         try {
-            BufferedReader reader = new BufferedReader(new FileReader(filename));
-            String line;
-            while ((line = reader.readLine()) != null) {
+            Scanner scanner = new Scanner(new File(filename));
+            while (scanner.hasNextLine()) {
+                String line = scanner.nextLine();
                 String[] words = line.split("\\s+");
                 for (String word : words) {
 
@@ -35,7 +34,7 @@
                 }
             }
 
-            reader.close();
+            scanner.close();
         } catch (Exception e) {
             e.printStackTrace();
         }

```


## Java 1.5: Final Look

```java
package org.example;

import java.io.File;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class ParallelWordCount implements Runnable {
    private static final int TOP_RESULTS = 10;
    private final String filename;
    private final Map<String, AtomicInteger> wordCount;

    public ParallelWordCount(String filename, Map<String, AtomicInteger> wordCount) {
        this.filename = filename;
        this.wordCount = wordCount;
    }

    public void run() {
        try {
            Scanner scanner = new Scanner(new File(filename));
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                String[] words = line.split("\\s+");
                for (String word : words) {

                    if (word.length() == 0)
                        continue;

                    wordCount.putIfAbsent(word, new AtomicInteger());
                    wordCount.get(word).getAndIncrement();
                }
            }

            scanner.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] fileNames) throws InterruptedException {
        final Map<String, AtomicInteger> totalWordCount = new ConcurrentHashMap<String, AtomicInteger>();
        ExecutorService executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

        for (String fileName : fileNames) {
            executorService.submit(new ParallelWordCount(fileName, totalWordCount));
        }

        executorService.shutdown();
        executorService.awaitTermination(60 * 5, TimeUnit.SECONDS);

        List<String> sortedWords = new ArrayList<String>(totalWordCount.keySet());
        Collections.sort(sortedWords, new Comparator<String>() {
            public int compare(String w1, String w2) {
                int count1 = totalWordCount.get(w1).intValue();
                int count2 = totalWordCount.get(w2).intValue();
                return count2 - count1;
            }
        });

        System.out.println("Top words:");
        for (String word : sortedWords.subList(0, TOP_RESULTS)) {
            int count = totalWordCount.get(word).get();

            System.out.println(word + ": " + count);
        }
    }
}
```

This particular use case alone is enough to demonstrate how much version 5 improved the java development experience.  


## Java 6: Nothing Significant

String classes isEmpty method can be used. TimeUnit.SECONDS can be replaced with TimeUnit.MINUTE.

```diff
Subject: [PATCH] isEmpty and minute
---
Index: src/main/java/org/example/ParallelWordCount.java
IDEA additional info:
Subsystem: com.intellij.openapi.diff.impl.patch.CharsetEP
<+>UTF-8
===================================================================
diff --git a/src/main/java/org/example/ParallelWordCount.java b/src/main/java/org/example/ParallelWordCount.java
--- a/src/main/java/org/example/ParallelWordCount.java	(revision 2b7baf095968c56a0e1d2711484379288b13a51a)
+++ b/src/main/java/org/example/ParallelWordCount.java	(revision d685cddff5e8a54746132efa4106d97ab37a659a)
@@ -26,7 +26,7 @@
                 String[] words = line.split("\\s+");
                 for (String word : words) {
 
-                    if (word.length() == 0)
+                    if (word.isEmpty())
                         continue;
 
                     wordCount.putIfAbsent(word, new AtomicInteger());
@@ -49,7 +49,7 @@
         }
 
         executorService.shutdown();
-        executorService.awaitTermination(60 * 5, TimeUnit.SECONDS);
+        executorService.awaitTermination(5, TimeUnit.MINUTES);
 
         List<String> sortedWords = new ArrayList<String>(totalWordCount.keySet());
         Collections.sort(sortedWords, new Comparator<String>() {


```

## Java 7: Try-with-resources

```diff
Subject: [PATCH] try-with-resource and diamond
---
Index: src/main/java/org/example/ParallelWordCount.java
IDEA additional info:
Subsystem: com.intellij.openapi.diff.impl.patch.CharsetEP
<+>UTF-8
===================================================================
diff --git a/src/main/java/org/example/ParallelWordCount.java b/src/main/java/org/example/ParallelWordCount.java
--- a/src/main/java/org/example/ParallelWordCount.java	(revision 77d6cca27fe5f49746f1daf2b07a4ac87f4b5e6b)
+++ b/src/main/java/org/example/ParallelWordCount.java	(revision c412708f6b91784ee90b2b942f739fada8327ee3)
@@ -19,8 +19,7 @@
     }
 
     public void run() {
-        try {
-            Scanner scanner = new Scanner(new File(filename));
+        try (Scanner scanner = new Scanner(new File(filename))) {
             while (scanner.hasNextLine()) {
                 String line = scanner.nextLine();
                 String[] words = line.split("\\s+");
@@ -34,14 +33,13 @@
                 }
             }
 
-            scanner.close();
         } catch (Exception e) {
             e.printStackTrace();
         }
     }
 
     public static void main(String[] fileNames) throws InterruptedException {
-        final Map<String, AtomicInteger> totalWordCount = new ConcurrentHashMap<String, AtomicInteger>();
+        final Map<String, AtomicInteger> totalWordCount = new ConcurrentHashMap<>();
         ExecutorService executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
 
         for (String fileName : fileNames) {
@@ -51,7 +49,7 @@
         executorService.shutdown();
         executorService.awaitTermination(5, TimeUnit.MINUTES);
 
-        List<String> sortedWords = new ArrayList<String>(totalWordCount.keySet());
+        List<String> sortedWords = new ArrayList<>(totalWordCount.keySet());
         Collections.sort(sortedWords, new Comparator<String>() {
             public int compare(String w1, String w2) {
                 int count1 = totalWordCount.get(w1).intValue();

```

## Java 8: Lambda Expressions and Streams API

Java 8 introduces two major concepts: Lambda functions and Streams API. Applying them changes the code substantially.

Note that putAbsent is replaced by computeIfAbsent which is more efficient and less verbose.

Note that fixed thread pool is replaced by work stealing pool. Work stealing pools allows threads to take over tasks from the queues of other threads if they are idle.

```diff
Subject: [PATCH] streams and lambda functions
---
Index: src/main/java/org/example/ParallelWordCount.java
IDEA additional info:
Subsystem: com.intellij.openapi.diff.impl.patch.CharsetEP
<+>UTF-8
===================================================================
diff --git a/src/main/java/org/example/ParallelWordCount.java b/src/main/java/org/example/ParallelWordCount.java
--- a/src/main/java/org/example/ParallelWordCount.java	(revision 2d79666ab26c558f2657f14d8a5b3348354846ea)
+++ b/src/main/java/org/example/ParallelWordCount.java	(revision 7ae13151bd5902f3c05062bed815320c07d25dd9)
@@ -7,6 +7,9 @@
 import java.util.concurrent.Executors;
 import java.util.concurrent.TimeUnit;
 import java.util.concurrent.atomic.AtomicInteger;
+import java.util.stream.Collectors;
+import java.util.stream.Stream;
+import java.util.stream.StreamSupport;
 
 public class ParallelWordCount implements Runnable {
     private static final int TOP_RESULTS = 10;
@@ -20,18 +23,11 @@
 
     public void run() {
         try (Scanner scanner = new Scanner(new File(filename))) {
-            while (scanner.hasNextLine()) {
-                String line = scanner.nextLine();
-                String[] words = line.split("\\s+");
-                for (String word : words) {
+            Stream<String> wordStream = StreamSupport.stream(Spliterators.spliteratorUnknownSize(scanner, Spliterator.ORDERED), false)
+                    .map(s -> s.split("\\s+")).flatMap(Arrays::stream);
 
-                    if (word.isEmpty())
-                        continue;
-
-                    wordCount.putIfAbsent(word, new AtomicInteger());
-                    wordCount.get(word).getAndIncrement();
-                }
-            }
+            wordStream.filter(word -> !word.isEmpty())
+                    .forEach(word -> wordCount.computeIfAbsent(word, (s) -> new AtomicInteger()).getAndIncrement());
 
         } catch (Exception e) {
             e.printStackTrace();
@@ -40,29 +36,17 @@
 
     public static void main(String[] fileNames) throws InterruptedException {
         final Map<String, AtomicInteger> totalWordCount = new ConcurrentHashMap<>();
-        ExecutorService executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
+        ExecutorService executorService = Executors.newWorkStealingPool(Runtime.getRuntime().availableProcessors());
 
-        for (String fileName : fileNames) {
-            executorService.submit(new ParallelWordCount(fileName, totalWordCount));
-        }
+        Arrays.stream(fileNames).map(fileName -> new ParallelWordCount(fileName, totalWordCount)).forEach(executorService::submit);
 
         executorService.shutdown();
         executorService.awaitTermination(5, TimeUnit.MINUTES);
 
-        List<String> sortedWords = new ArrayList<>(totalWordCount.keySet());
-        Collections.sort(sortedWords, new Comparator<String>() {
-            public int compare(String w1, String w2) {
-                int count1 = totalWordCount.get(w1).intValue();
-                int count2 = totalWordCount.get(w2).intValue();
-                return count2 - count1;
-            }
-        });
-
         System.out.println("Top words:");
-        for (String word : sortedWords.subList(0, TOP_RESULTS)) {
-            int count = totalWordCount.get(word).get();
-
-            System.out.println(word + ": " + count);
-        }
+        totalWordCount.entrySet().stream().collect(Collectors.toMap(Map.Entry::getKey, e -> e.getValue().intValue())).entrySet().stream()
+                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
+                .limit(TOP_RESULTS)
+                .forEach(s -> System.out.println(s.getKey() + ": " + s.getValue()));
     }
 }

```

## Java 8: Final Look

Using features introduced in Java 8, makes the code more concise and readable. 

Comparing Java 1.4 code and Java 8 code reveals the fact that modern Java is quiet different from the language used in early 2000s. 

```java
package org.example;

import java.io.File;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

public class ParallelWordCount implements Runnable {
    private static final int TOP_RESULTS = 10;
    private final String filename;
    private final Map<String, AtomicInteger> wordCount;

    public ParallelWordCount(String filename, Map<String, AtomicInteger> wordCount) {
        this.filename = filename;
        this.wordCount = wordCount;
    }

    public void run() {
        try (Scanner scanner = new Scanner(new File(filename))) {
            Stream<String> wordStream = StreamSupport.stream(Spliterators.spliteratorUnknownSize(scanner, Spliterator.ORDERED), false)
                    .map(s -> s.split("\\s+")).flatMap(Arrays::stream);

            wordStream.filter(word -> !word.isEmpty())
                    .forEach(word -> wordCount.computeIfAbsent(word, (s) -> new AtomicInteger()).getAndIncrement());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] fileNames) throws InterruptedException {
        final Map<String, AtomicInteger> totalWordCount = new ConcurrentHashMap<>();
        ExecutorService executorService = Executors.newWorkStealingPool(Runtime.getRuntime().availableProcessors());

        Arrays.stream(fileNames).map(fileName -> new ParallelWordCount(fileName, totalWordCount)).forEach(executorService::submit);

        executorService.shutdown();
        executorService.awaitTermination(5, TimeUnit.MINUTES);

        System.out.println("Top words:");
        totalWordCount.entrySet().stream().collect(Collectors.toMap(Map.Entry::getKey, e -> e.getValue().intValue())).entrySet().stream()
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                .limit(TOP_RESULTS)
                .forEach(s -> System.out.println(s.getKey() + ": " + s.getValue()));
    }
}
```

## Conclusion 
   
Java is a popular programming language that has been evolving over time to meet the changing needs of developers. The introduction of new features and changes has not always been regular or timely, but they have always come at the right moments to keep Java popular.

Java 5 was a major milestone in the history of Java. It introduced a number of new features, including generics, annotations, and autoboxing. These features made Java more powerful and flexible, and they helped to make it a more popular choice for developers.

Java 8 was another major milestone. It introduced a number of new features, including lambda expressions, streams, and default methods. These features made Java more concise and expressive, and they helped to make it a more popular choice for developers.

In this article we tried to demonstrate evolution of Java using a simple use case. In fact this article can only offer a gist of how much Java evolved over the time. 


## References
1. [Java 5 Docs](https://docs.oracle.com/javase/1.5.0/docs/api/)
2. [Java 6 Docs](https://docs.oracle.com/javase/6/docs/api/)
3. [Java 7 Docs](https://docs.oracle.com/javase/7/docs/api/)
4. [Java 8 Docs](https://docs.oracle.com/javase/8/docs/api/)
5. [Version History](https://en.wikipedia.org/wiki/Java_version_history)