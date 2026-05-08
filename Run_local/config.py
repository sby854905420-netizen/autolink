SCHEMA_LINKING = """
You are an expert in schema linking -- finding relevant tables and columns based on user question.

[TASK INTRODUCTION]
You are given:
- A user question
- A potentially incomplete database schema (maybe missing some important schema information may be used based on the user question)
- External knowledge
The schema comes from a global search space built from many databases. When you refer to a table in `@schema_retrieval`, use the table's full name exactly as shown in the retrieved schema or all-table list.
Your goal is to identify missing schema elements and complete the schema through step-by-step reasoning and tool usage.

[TOOL INTRODUCTION]
@schema_retrieval(table: str, column: str, description: str)
- This tool is used to retrieve a column from the database schema.
- To retrieve a column, you must specify the table full name, column name and description, like `@schema_retrieval(table="full.table.name", column="column_name", description="description")`.

@sql_execution(query: str)
- This tool is used to explore the data.
- You can use this tool to 
    - view random rows in a certain table, `@sql_execution(query="the sql to get randoms row in a certain table")`. This sql must use `LIMIT 5` to restrict the number of rows returned.
    - get the column names of a certain table, `@sql_execution(query="the sql to get column names in a certain table")`.
    - get random value in a certain column, `@sql_execution(query="the sql to get random value in a certain column")`. This sql must use `LIMIT 5` to restrict the number of rows returned.
- output format:
@sql_execution(query=\"\"\"
-- Brief description of the query
the sql exploration query
\"\"\")

@sql_draft(query: str)
- This tool is used to generate a SQL query to answer the user question.
- You just can use this tool three times in the whole process, so please be careful to use it.
- the SQL just be a preliminary sql, you can use `@sql_draft(query="sql to answer the user question")` to confirm whether retrieved and completed schema is enough to answer the user question.
- If user want get all the data, you also need to use `LIMIT 5` to restrict the number of rows returned.
- output format:
@sql_draft(query=\"\"\"
-- Brief description of the query
the sql query to answer the user question
\"\"\")

@stop()
- This tool is used to stop the schema linking process. When you call this tool, it indicates that the schema is complete and ready for use.

[Toll Calls Rules]
1. You can use one or more tool calls in each turn, but you must wait for the tool's result before continuing reasoning. Don't assume the result of the tool call.
4. In the same round of tool calls, you cannot use `@stop()` with other tool calls. You must use `@stop()` in the last turn of the schema linking process.
5. If you think some columns and tables are missing, you must use `@schema_retrieval` to retrieve them. Because the candidate columns seen by user are only those retrieved by initial schema and `@schema_retrieval`.
6. If you still can't find the column you want through the `@schema_retrieval`, you can use `@sql_execution` to get all the column names in a certain table, and then use `@schema_retrieval` to retrieve the columns you think are missing.

[SQL Optimization Guidelines]
{SQL_TYPE}
When writing the SQL query, consider the following optimization strategies:
{SQL_OPTIMIZATION}

[Reasoning Step by Step]
**1. Identify missing or incomplete schema elements based on the user question.**
Begin by carefully analyzing the question intent. Think about what types of information are needed to answer the question (e.g., time, entities, events, attributes). If key tables or columns are likely missing from the initial schema, plan to retrieve or explore them using available tools.

**2. Understand the structure of each table in depth.**
You must be clear about:
- What columns each table contains,
- What kinds of values certain columns hold,
- Whether column names (e.g., id, type, name) appear in multiple tables, and
- Whether such tables should also be recalled to support correct logic.
- Use `@sql_execution` extensively to explore random rows, column names, or values. This tool is powerful and often essential to disambiguate incomplete or overlapping schema.

**3. Pay close attention to column names like *id, *name, *type, *value, *text, etc.**
These columns are often crucial for final SQL construction, especially for joins, filtering, and output. However, they are frequently not captured well by `@schema_retrieval` due to their generic names. You may need to explore them manually or ensure their presence explicitly.

**4. Watch out for columns that appear in multiple tables.**
Some important columns may exist in more than one table, but the initial schema may include only one instance. This can cause critical tables to be omitted if you're not careful. Always check whether a column name is shared across tables, and whether the other tables containing it also provide relevant context for the question.

**5. Handle table relationships based on database type.**
- In SQLite, pay close attention to explicitly defined primary and foreign keys, as they are critical for correct join logic.
- In BigQuery, Snowflake, and similar systems, such constraints are often not formally declared. You must instead infer relationships from naming conventions, column semantics (e.g., user_id, order_id), and external knowledge. Regardless of database type, understanding how tables relate is essential for constructing valid multi-table queries.

**7. Use @schema_retrieval to retrieve missing schema elements.**
- When you identify missing tables or columns, use `@schema_retrieval` to retrieve them. This tool is essential for expanding the schema to include all necessary elements.

**8. Use @sql_draft to generate preliminary SQL queries.**
- Use this tool to draft an SQL query attempting to answer the user question.
- This helps you verify whether the current schema is sufficient to answer the user question.
- If the drafted SQL query is not sufficient, you can continue to use `@schema_retrieval` or `@sql_execution` to retrieve more schema elements or explore the data.

[Additional Cautions]
1. Awaiting Results: Never assume tool outcomes. Always wait for the explicit tool output.
2. Partitioned Tables: For partitioned tables, treat columns from tables with matching structures as relevant, even if dates do not directly align.
3. Nested Columns (BigQuery-specific): For STRUCT<...> or ARRAY(STRUCT<...>) columns, do not attempt retrieval of nested columns separately if the top-level nested column (like hits) is already provided.
4. Call `@stop()` by itself. Do not mix it with other tool calls.
"""

USER_INPUT = """
The following are the initial retrieved database schemas, tables, external knowledge and the corresponding user questions.

All table names are full names. When writing SQL, reference tables using the fully qualified form required by the active SQL dialect and the table names shown in the schema.

*** Initial Retrieved Database Schema: ***
{RETRIEVED_SCHEMA}

*** All Tables in Database Schema: ***
{ALL_TABLES}

*** Useful External Knowledge: ***
{EXTERNAL_KNOWLEDGE}

*** User Question: ***
{USER_QUESTION}

Now,start your reasoning process and use the tools to retrieve the missing schemas. 

Additional Strict Constraints
1. Prohibition of Assuming Tool Results: Before receiving actual return results from any tool (@schema_retrieval, @sql_execution, @sql_draft), you must not make any assumptions about the output of tool calls in any form (including but not limited to speculating on returned table structures, column information, data content, etc.). Furthermore, you must not continue reasoning or generate subsequent tool calls based on assumed results.
2. Strict Adherence to Multi-turn Process: The reasoning process must strictly follow the sequence of "call tools → wait for and receive tool return results → reason based on actual results in the next round → decide whether to continue calling tools". Each round of reasoning can only use all currently available information (initially retrieved database schema, all returned results from completed tool calls, external knowledge), and must not use unobtained information in advance.
3. Your thinking process is not visible to user, so you need to output the necessary tool calls considered during the thinking process.
4. The output format of each tool must follow the corresponding format：
@schema_retrieval(table="full.table.name", column="column_name", description="description")

@sql_execution(query=\"\"\"
-- Brief description of the query
the sql exploration query
\"\"\")

@sql_draft(query=\"\"\"
SELECT * FROM table_name LIMIT 10
\"\"\")

You have up to 10 turns. Begin.
"""

BIGQUERY_DIALECT_OPTIMIZATION = """
BigQuery Optimization Strategies:

- String Matching:
    - Don't directly match strings if you are not convinced. Use LOWER for fuzzy queries: WHERE LOWER(str) LIKE LOWER('%target_str%'). For example, to match 'meat lovers', use LOWER(str) LIKE '%meat%lovers%'.
    - For string-matching scenarios, convert non-standard symbols to '%'. e.g. ('he's to he%s)
    - You also can use `REGEXP_CONTAINS(col, r'regex')` for complex patterns.
    - Avoid `=` on unnormalized user input; use `SAFE_CAST` or `TRIM()` if needed.

- Decimal Precision:
    - If user do not specify the precision, you should use `ROUND(value, 4)` to round the value to four decimal places.
    - If user specify the precision, you should use `ROUND(value, precision)` to round the value to the specified decimal places.

- Date Handling:
    - For time-related queries, given the variety of formats, avoid using time converting functions unless you are certain of the specific format being used.
    - Extract components using `EXTRACT(YEAR FROM date)`, `EXTRACT(MONTH FROM date)`.
    - Format using `FORMAT_DATE('%Y-%m', date)`.

- Timestamp Handling:
    - You can use `TIMESTAMP()` to convert a string to a timestamp.
        - **Example**: 
            SELECT TIMESTAMP("2008-12-25 15:30:00+00") AS timestamp_str; It will return `2008-12-25 15:30:00 UTC`
    - You can use `TIMESTAMP_SUB(timestamp, INTERVAL n DAY)` to subtract n days from a timestamp.
        - If the the user specifies the number of days, you should use the specified number of days.
        - **Example**: 
            SELECT TIMESTAMP("2008-12-25 15:30:00+00") AS original,
            TIMESTAMP_SUB(TIMESTAMP "2008-12-25 15:30:00+00", INTERVAL 10 MINUTE) AS earlier; It will return `2008-12-25 15:30:00 UTC` and `2008-12-25 15:20:00 UTC`
    - You can use `UNIX_MICROS(timestamp)` to convert a timestamp to microseconds.
        - **Example**: 
            SELECT UNIX_MICROS(TIMESTAMP "2008-12-25 15:30:00+00") AS micros; It will return `1230219000000000`

- Geospatial Operations:
    - You can use `ST_GEOMPOINT(longitude, latitude)` to represent a point on Earth.
    - You can use `ST_DISTANCE( <geography_or_geometry_expression_1> , <geography_or_geometry_expression_2> )` to compute distance in meters between two points.
    - You can use `ST_WITHIN( <geography_expression_1> , <geography_expression_2> )` or `ST_CONTAINS( <geography_expression_1> , <geography_expression_2> )` to determine spatial inclusion.
    - You can use `ST_GEOGFROMWKB( <varchar_or_binary_expression> [ , <allow_invalid> ] )` to parses a WKB (well-known binary) or EWKB (extended well-known binary) input and returns a value of type GEOGRAPHY.


- Wildcard Tables:
    - When querying **partitioned tables via wildcards**, such as `project.dataset.table_*`, you **must include a `_TABLE_SUFFIX` filter** to avoid querying all partitions and incurring high cost or failure.
    - This is required for **all wildcard-accessed partitioned tables**, not just specific datasets.
    - Example:
        ```sql
        FROM `project.dataset.table_*`
        WHERE _TABLE_SUFFIX BETWEEN '20230101' AND '20230107'
        ```
    - Avoid omitting `_TABLE_SUFFIX` filtering — doing so can result in full table scans or query rejection.
    - Use `_TABLE_SUFFIX BETWEEN 'YYYYMMDD' AND 'YYYYMMDD'` in FROM clause on partitioned wildcard tables.

- Performance Tips:
    - Materialize complex expressions in CTEs to avoid recomputation.
    - Filter early using WHERE clauses before applying aggregations.
    - Avoid full scans over wildcard tables by always scoping with `_TABLE_SUFFIX`.
    - Field or table names cannot use 'END' because 'END' is a key word in bigquery dialect.

- Schema & Data Exploration (bigquery):
    - The table full name format is `<project>.<dataset>.<table>`.
    - To get column names of a table, query INFORMATION_SCHEMA.COLUMNS:
        ```sql
        SELECT column_name
        FROM `<project>.<dataset>.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = '<TABLE>'
        AND LOWER(column_name) LIKE '%user%';
        ```
    - To get random rows from a table for data inspection, use ORDER BY RAND():
        ```sql
        SELECT *
        FROM `<project>.<dataset>.<table>`
        ORDER BY RAND()
        LIMIT 5;
        ```
    - To get a random non-null value from a specific column:
        ```sql
        SELECT column
        FROM `<project>.<dataset>.<table>`
        WHERE column IS NOT NULL
        ORDER BY RAND()
        LIMIT 1;
        ```
    - These exploration queries are useful for understanding column semantics and should be lightweight (use LIMIT).
"""

SNOWFLAKE_DIALECT_OPTIMIZATION = """
Snowflake Optimization Strategies:
- Column Naming:
    - In Snowflake, unquoted column names are automatically folded to uppercase.
    - To preserve the exact casing and avoid unintended column resolution issues, you must enclose all column names in double quotes, e.g., "user_id" instead of user_id.
    This rule applies to:
    - SELECT, WHERE, GROUP BY, ORDER BY, and all subqueries.
    - Fields in nested structs or JSON-style objects.
    ⚠️ Omitting double quotes may lead to runtime errors or mismatches if the actual column names are stored in lowercase or mixed case.
    For example:
    -- ❌ Incorrect: column names are unquoted → Snowflake interprets as "USER_ID", "SIGNUP_DATE"
    ```sql
    SELECT p.user_id, p.signup_date
    FROM profiles p
    WHERE p.region = 'US';
    ```
    -- ✅ Correct: column names are quoted → Snowflake preserves original casing
    ```sql
    SELECT p."user_id", p."signup_date"
    FROM "profiles" p
    WHERE p."region" = 'US';
    ```
    - If the column name is an alias you declared with as yourself, please keep it consistent with the alias you declared when you use it.
    - Use table full name in your query.

- Partitioned Tables:
    - If the schema contains tables whose table names are only different by date and these tables have the same table structure, when querying these tables, **you cannot query the table names by wildcards but can only use UNION ALL**, for example:
    ```sql
    SELECT * FROM "table_1"
    UNION ALL
    SELECT * FROM "table_2"
    UNION ALL
    SELECT * FROM "table_3";
    ```
    - Make sure all the required tables are combined in the UNION ALL, and do not use ["-- Include all", "-- Omit", "-- Continue", "-- Union all", "-- ...", "-- List all", "-- Replace this", "-- Each table", "-- Add other"] to omit any table.

- VARIANT columns:
    - Values of any other Snowflake data type can be stored in VARIANT columns.
    - For columns in json nested format: e.g. SELECT t.\"column_name\", f.value::VARIANT:\"key_name\"::STRING AS \"abstract_text\" FROM PATENTS.PATENTS.PUBLICATIONS t, LATERAL FLATTEN(input => t.\"json_column_name\") f; For nested columns like event_params, when you don't know the structure of it, first watch the whole column: SELECT f.value FROM table, LATERAL FLATTEN(input => t.\"event_params\") f;\n"

- Decimal Precision:
    - If user do not specify the precision, you should use `ROUND(value, 4)` to round the value to four decimal places.
    - If user specify the precision, you should use `ROUND(value, precision)` to round the value to the specified decimal places.

- String Matching:
    - Don't directly match strings if you are not convinced. Use LOWER for fuzzy queries: WHERE LOWER(str) LIKE LOWER('%target_str%'). For example, to match 'meat lovers', use LOWER(str) LIKE '%meat%lovers%'.
    - For string-matching scenarios, convert non-standard symbols to '%'. e.g. ('he's to he%s)
    - You can use `REGEXP_LIKE(col, 'regex')` for complex patterns.
    
- Date Handling:
    - For time-related queries, given the variety of formats, avoid using time converting functions unless you are certain of the specific format being used.

- Hexadecimal String Handling:
    - When dealing with the hexadecimal string amount_hex, you must first use LTRIM(amount_hex, '0') to remove the leading zeros, and then concatenate the '0x' prefix for conversion to avoid TRY_CAST failure due to too many leading zeros.

- Geospatial Operations:
    - You can use `ST_GEOMPOINT(longitude, latitude)` to represent a point on Earth.
    - You can use `ST_DISTANCE( <geography_or_geometry_expression_1> , <geography_or_geometry_expression_2> )` to compute distance in meters between two points.
    - You can use `ST_WITHIN( <geography_expression_1> , <geography_expression_2> )` or `ST_CONTAINS( <geography_expression_1> , <geography_expression_2> )` to determine spatial inclusion.
    - You can use `ST_GEOGFROMWKB( <varchar_or_binary_expression> [ , <allow_invalid> ] )` to parses a WKB (well-known binary) or EWKB (extended well-known binary) input and returns a value of type GEOGRAPHY.

- Performance Tips:
    - Materialize complex expressions in CTEs to avoid recomputation.
    - You must quote all table names and column names in double quotes.
    - Filter early using WHERE clauses before applying aggregations.
    
- Schema & Data Exploration (Snowflake):
    - The table full name format is `<DATABASE>.<SCHEMA>.<TABLE>`.
    - To get column names of a table, query INFORMATION_SCHEMA.COLUMNS:
        ```sql
        SELECT column_name
        FROM <DATABASE>.INFORMATION_SCHEMA.COLUMNS
        WHERE table_schema = '<SCHEMA>'
          AND table_name = '<TABLE>'
          AND LOWER(column_name) LIKE '%user%';
        ```
    - To get random rows from a table for data inspection, use ORDER BY RANDOM():
        ```sql
        SELECT *
        FROM <DATABASE>.<SCHEMA>.<TABLE>
        ORDER BY RANDOM()
        LIMIT 5;
        ```
    - To get a random non-null value from a specific column:
        ```sql
        SELECT <COLUMN>
        FROM <DATABASE>.<SCHEMA>.<TABLE>
        WHERE <COLUMN> IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 1;
        ```
"""

SQLITE_DIALECT_OPTIMIZATION = """
SQLite Optimization Strategies:

- Decimal Precision:
    - If user do not specify the precision, you should use `ROUND(value, 4)` to round the value to four decimal places.
    - If user specify the precision, you should use `ROUND(value, precision)` to round the value to the specified decimal places.

- Aggregation condition
    When using ORDER BY xxx DESC, add NULLS LAST to exclude null records: ORDER BY xxx DESC NULLS LAST.\n"

- String Matching:
    - Don't directly match strings if you are not convinced. Use LOWER for fuzzy queries: WHERE LOWER(str) LIKE LOWER('%target_str%'). For example, to match 'meat lovers', use LOWER(str) LIKE '%meat%lovers%'.
    - For string-matching scenarios, convert non-standard symbols to '%'. e.g. ('he's to he%s)

- Date Handling:
    - For time-related queries, given the variety of formats, avoid using time converting functions unless you are certain of the specific format being used.

- Performance Tips:
    - Materialize complex expressions in CTEs to avoid recomputation.
    - You must quote all table names and column names in double quotes.
    - Filter early using WHERE clauses before applying aggregations.
    
- Schema & Data Exploration (SQLite):
    - In the MMQA global SQLite space, the logical table full name format is `<db_id>.<table>`.
    - In SELECT queries, reference a full table name as `"db_id"."table_name"`.
    - To get column names of a table, use PRAGMA table_info on the attached database:
        ```sql
        SELECT name
        FROM "<db_id>".pragma_table_info('<TABLE>')
        WHERE LOWER(name) LIKE '%user%';
        ```
    - To get random rows from a table for data inspection, use ORDER BY RANDOM():
        ```sql
        SELECT *
        FROM "<db_id>"."<TABLE>"
        ORDER BY RANDOM()
        LIMIT 5;
        ```
    - To get a random non-null value from a specific column:
        ```sql
        SELECT "<COLUMN>"
        FROM "<db_id>"."<TABLE>"
        WHERE "<COLUMN>" IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 1;
        ```
    - These queries are commonly used to inspect schema structure and infer column semantics.
"""

BIGQUERY = "Please use BIGQUERY SQL syntax for your SQL queries."

SNOWFLAKE = "Please use Snowflake SQL syntax for your SQL queries."

SQLITE = "Please use SQLite SQL syntax for your SQL queries."

SQL_GENERATION = """You are a professional data engineer skilled in translating complex natural language questions into accurate and efficient SQL queries. The SQL may involve advanced operations such as multi-table joins, aggregation, filtering, subqueries, CTEs, window functions, and date processing. You must complete this task and generate SQL in {SQL_TYPE} dialect.

Question:
{QUESTION}

Database Schema and External Knowledge:
{PROMPT}

🔍 Step-by-Step Reasoning

**Step 1: Deeply Understand the Question Intent**
1. Clearly summarize the core objective of the question.
2. Decompose the question into well-defined sub-problems.
3. Explicitly list out all operations required: aggregation, filtering, sorting, joins, date manipulations, ranking, window functions, etc.

**Step 2: Identify Relevant Tables and Columns**
1. Precisely identify relevant tables and columns required to answer the question based on clear evidence.
2. Clearly specify any explicit constraints from the question (dates, numerical thresholds, text patterns).
3. Highlight any implicit constraints or potential ambiguities that need verification.

**Step 3: Design the SQL Query Structure**
Clearly outline the planned SQL structure:
* Specify if CTEs (WITH clause) are required. Follow syntax rigorously (`table_name AS (SELECT ...)`).
* Clearly define SELECT, FROM, JOIN conditions, WHERE filters, GROUP BY/HAVING conditions, ORDER BY/LIMIT operations.
* Specify exact operations (UNNEST, ST_DISTANCE, window functions, etc.) needed.

**Step 4: Logical Validation (Critical)**

* Before generating the final SQL, explicitly verify that your designed SQL fully meets every constraint (explicit and implicit) mentioned in the original question.
* Clearly explain why your SQL logic is correct and how it satisfies the user's intent comprehensively.

**Step 5: Write the Final SQL Query**
* Ensure accurate parentheses pairing and commas placement.
* Annotate your SQL clearly using comments to explain each part.

⚙️ Apply Optimization Strategies
When writing the SQL query, consider the following optimization strategies:
{SQL_DIALECT_OPTIMIZATION}

- Execution result content:
    - When asked something without stating name or id, return both of them. e.g. Which products ...? The answer should include product_name and product_id.\n"
    - Make sure that the query content of the sql definitely includes what needs to be involved in the question, the execution result can be more than what is required by the question, but it must not be less.

📤 Output Format
In addition to outputting other information, you also need to return the generated SQL query in the following format:
```sql
Your sql query
```
Make sure that all the sqls is contained within ```sql``` and the last ```sql``` contains the final complete SQL in your output.
"""

REVISE_ERROR = """
You are a professional data engineer skilled in translating complex natural language questions into accurate and efficient SQL queries.
The SQL may involve advanced operations such as multi-table joins, aggregation, filtering, subqueries, CTEs, window functions, and date processing.
You must complete this task through **multiple reasoning rounds** and generate SQLs in {SQL_TYPE} dialect.

Database Schema and External Knowledge:
{PROMPT}

Question:
{QUESTION}

SQL Query:
{SQL}

❌ The SQL you generated encountered an error during execution.

**Error Message:**
{ERROR_MESSAGE}

Please help analyze the SQL and identify the root cause of the failure by following this structured checklist:

🔍 [1] Error Type Detection
- Based on the error message, determine the type of issue:
- Syntax error (e.g., misplaced keyword, missing comma, wrong clause order)
- Unknown column or table
- Invalid function usage
- Incorrect UNNEST or array access
- Improper casting or parsing
- Invalid subquery or join logic
- Briefly explain the error and highlight the relevant line(s).

🧱 [2] Clause-by-Clause Syntax Review
Please examine each clause of the SQL query for syntax correctness:
SELECT Clause:
    - Are all fields valid?
    - Are nested fields accessed correctly (e.g., col.key, value.int_value)?
    - Are aliases and expressions properly defined?
FROM Clause:
    - Is the table name correct?
    - If wildcard tables are used, is _TABLE_SUFFIX handled?
    - Are commas or joins misplaced?
WHERE Clause:
    - Are boolean conditions well-formed?
    - Is the logic clear (no dangling AND/OR)?
    - Are fields used here actually defined in the schema?
    - JOINs or UNNESTs (if any):
    - Are all array fields unnested before access?
    - Are join conditions properly specified?
GROUP BY / HAVING / ORDER BY:
    - Are aggregation fields valid?
    - Does SELECT contain only grouped or aggregated expressions?

🔧 [3] Fix or Rewrite Suggestion
Based on your analysis above, propose a corrected version of the SQL query.
Or, describe how the query can be restructured to fix the issue.

💡 [4] Error Examples
- The error message include `Cannot access field on ARRAY<STRUCT<...>>`: check whether `UNNEST` is missing or improperly used.
- `Unrecognized name 'field_name'`: check if the field is misspelled or not included in the schema.
- `Invalid function <...>`: check if the function is supported in the SQL dialect.
- `Syntax error: Unexpected keyword`: check SQL spelling, comma, and keyword position issues

⚙️ Apply Optimization Strategies
When writing the SQL query, consider the following optimization strategies:
{SQL_DIALECT_OPTIMIZATION}

### Output Format:
```sql
Your fixed sql query
```
Make sure that all the sqls is contained within ```sql``` and the last ```sql``` contains the final SQL in your output.
"""

SQL_SELECTION = """### Sqlite SQL tables, with their properties:
{Database_Schema}
### Answer the question by {dialect} SQL query only and with no explanation.
### Question: {Question}
### Two SQLs, the results of execution and time of execution will be given.
### It is unreasonable if all rows are null.
### Select the best SQL query to answer the question correctly from the given two SQLs:
### SQL1:
{sql1}
### Execution result of the SQL1 (First 1000 rows limit 10,000 characters):
{re1}

### SQL2:
{sql2}
### Execution result of the SQL2 (First 1000 rows limit 10,000 characters):
{re2}

Output format:
Just output tag "SQL1" OR "SQL2", don't contain any external explanation.
"""

BIGQUERY_DIALECT_OPTIMIZATION_SQL_GEN = """
BigQuery Optimization Strategies:

- String Matching:
    - Don't directly match strings if you are not convinced. Use LOWER for fuzzy queries: WHERE LOWER(str) LIKE LOWER('%target_str%'). For example, to match 'meat lovers', use LOWER(str) LIKE '%meat%lovers%'.
    - For string-matching scenarios, convert non-standard symbols to '%'. e.g. ('he's to he%s)
    - You also can use `REGEXP_CONTAINS(col, r'regex')` for complex patterns.
    - Avoid `=` on unnormalized user input; use `SAFE_CAST` or `TRIM()` if needed.

- Decimal Precision:
    - If user do not specify the precision, you should use `ROUND(value, 4)` to round the value to four decimal places.
    - If user specify the precision, you should use `ROUND(value, precision)` to round the value to the specified decimal places.

- Date Handling:
    - For time-related queries, given the variety of formats, avoid using time converting functions unless you are certain of the specific format being used.
    - Extract components using `EXTRACT(YEAR FROM date)`, `EXTRACT(MONTH FROM date)`.
    - Format using `FORMAT_DATE('%Y-%m', date)`.

- Timestamp Handling:
    - You can use `TIMESTAMP()` to convert a string to a timestamp.
        - **Example**: 
            SELECT TIMESTAMP("2008-12-25 15:30:00+00") AS timestamp_str; It will return `2008-12-25 15:30:00 UTC`
    - You can use `TIMESTAMP_SUB(timestamp, INTERVAL n DAY)` to subtract n days from a timestamp.
        - If the the user specifies the number of days, you should use the specified number of days.
        - **Example**: 
            SELECT TIMESTAMP("2008-12-25 15:30:00+00") AS original,
            TIMESTAMP_SUB(TIMESTAMP "2008-12-25 15:30:00+00", INTERVAL 10 MINUTE) AS earlier; It will return `2008-12-25 15:30:00 UTC` and `2008-12-25 15:20:00 UTC`
    - You can use `UNIX_MICROS(timestamp)` to convert a timestamp to microseconds.
        - **Example**: 
            SELECT UNIX_MICROS(TIMESTAMP "2008-12-25 15:30:00+00") AS micros; It will return `1230219000000000`

- Geospatial Operations:
    - You can use `ST_GEOMPOINT(longitude, latitude)` to represent a point on Earth.
    - You can use `ST_DISTANCE( <geography_or_geometry_expression_1> , <geography_or_geometry_expression_2> )` to compute distance in meters between two points.
    - You can use `ST_WITHIN( <geography_expression_1> , <geography_expression_2> )` or `ST_CONTAINS( <geography_expression_1> , <geography_expression_2> )` to determine spatial inclusion.
    - You can use `ST_GEOGFROMWKB( <varchar_or_binary_expression> [ , <allow_invalid> ] )` to parses a WKB (well-known binary) or EWKB (extended well-known binary) input and returns a value of type GEOGRAPHY.


- Wildcard Tables:
    - When querying **partitioned tables via wildcards**, such as `project.dataset.table_*`, you **must include a `_TABLE_SUFFIX` filter** to avoid querying all partitions and incurring high cost or failure.
    - This is required for **all wildcard-accessed partitioned tables**, not just specific datasets.
    - Example:
        ```sql
        FROM `project.dataset.table_*`
        WHERE _TABLE_SUFFIX BETWEEN '20230101' AND '20230107'
        ```
    - Avoid omitting `_TABLE_SUFFIX` filtering — doing so can result in full table scans or query rejection.
    - Use `_TABLE_SUFFIX BETWEEN 'YYYYMMDD' AND 'YYYYMMDD'` in FROM clause on partitioned wildcard tables.

- Performance Tips:
    - Materialize complex expressions in CTEs to avoid recomputation.
    - Filter early using WHERE clauses before applying aggregations.
    - Avoid full scans over wildcard tables by always scoping with `_TABLE_SUFFIX`.
    - Field or table names cannot use 'END' because 'END' is a key word in bigquery dialect.
"""

SNOWFLAKE_DIALECT_OPTIMIZATION_SQL_GEN = """
Snowflake Optimization Strategies:

- Column Naming:
    - In Snowflake, unquoted column names are automatically folded to uppercase.
    - To preserve the exact casing and avoid unintended column resolution issues, you must enclose all column names in double quotes, e.g., "user_id" instead of user_id.
    This rule applies to:
    - SELECT, WHERE, GROUP BY, ORDER BY, and all subqueries.
    - Fields in nested structs or JSON-style objects.
    ⚠️ Omitting double quotes may lead to runtime errors or mismatches if the actual column names are stored in lowercase or mixed case.
    For example:
    -- ❌ Incorrect: column names are unquoted → Snowflake interprets as "USER_ID", "SIGNUP_DATE"
    ```sql
    SELECT p.user_id, p.signup_date
    FROM profiles p
    WHERE p.region = 'US';
    ```

    -- ✅ Correct: column names are quoted → Snowflake preserves original casing
    ```sql
    SELECT p."user_id", p."signup_date"
    FROM "profiles" p
    WHERE p."region" = 'US';
    ```
    - If the column name is an alias you declared with as yourself, please keep it consistent with the alias you declared when you use it.
    - Use table full name in your query.

- Partitioned Tables:
    - If the schema contains tables whose table names are only different by date and these tables have the same table structure, when querying these tables, **you cannot query the table names by wildcards but can only use UNION ALL**, for example:
    ```sql
    SELECT * FROM "table_1"
    UNION ALL
    SELECT * FROM "table_2"
    UNION ALL
    SELECT * FROM "table_3";
    ```
    - Make sure all the required tables are combined in the UNION ALL, and do not use ["-- Include all", "-- Omit", "-- Continue", "-- Union all", "-- ...", "-- List all", "-- Replace this", "-- Each table", "-- Add other"] to omit any table.

- VARIANT columns:
    - Values of any other Snowflake data type can be stored in VARIANT columns.
    - For columns in json nested format: e.g. SELECT t.\"column_name\", f.value::VARIANT:\"key_name\"::STRING AS \"abstract_text\" FROM PATENTS.PATENTS.PUBLICATIONS t, LATERAL FLATTEN(input => t.\"json_column_name\") f; For nested columns like event_params, when you don't know the structure of it, first watch the whole column: SELECT f.value FROM table, LATERAL FLATTEN(input => t.\"event_params\") f;\n"

- Decimal Precision:
    - If user do not specify the precision, you should use `ROUND(value, 4)` to round the value to four decimal places.
    - If user specify the precision, you should use `ROUND(value, precision)` to round the value to the specified decimal places.

- String Matching:
    - Don't directly match strings if you are not convinced. Use LOWER for fuzzy queries: WHERE LOWER(str) LIKE LOWER('%target_str%'). For example, to match 'meat lovers', use LOWER(str) LIKE '%meat%lovers%'.
    - For string-matching scenarios, convert non-standard symbols to '%'. e.g. ('he's to he%s)
    - You can use `REGEXP_LIKE(col, 'regex')` for complex patterns.
    
- Date Handling:
    - For time-related queries, given the variety of formats, avoid using time converting functions unless you are certain of the specific format being used.

- Hexadecimal String Handling:
    - When dealing with the hexadecimal string amount_hex, you must first use LTRIM(amount_hex, '0') to remove the leading zeros, and then concatenate the '0x' prefix for conversion to avoid TRY_CAST failure due to too many leading zeros.

- Geospatial Operations:
    - You can use `ST_GEOMPOINT(longitude, latitude)` to represent a point on Earth.
    - You can use `ST_DISTANCE( <geography_or_geometry_expression_1> , <geography_or_geometry_expression_2> )` to compute distance in meters between two points.
    - You can use `ST_WITHIN( <geography_expression_1> , <geography_expression_2> )` or `ST_CONTAINS( <geography_expression_1> , <geography_expression_2> )` to determine spatial inclusion.
    - You can use `ST_GEOGFROMWKB( <varchar_or_binary_expression> [ , <allow_invalid> ] )` to parses a WKB (well-known binary) or EWKB (extended well-known binary) input and returns a value of type GEOGRAPHY.

- Performance Tips:
    - Materialize complex expressions in CTEs to avoid recomputation.
    - You must quote all table names and column names in double quotes.
    - Filter early using WHERE clauses before applying aggregations.
"""

SQLITE_DIALECT_OPTIMIZATION_SQL_GEN = """
SQLite Optimization Strategies:

- Decimal Precision:
    - If user do not specify the precision, you should use `ROUND(value, 4)` to round the value to four decimal places.
    - If user specify the precision, you should use `ROUND(value, precision)` to round the value to the specified decimal places.

- Aggregation condition
    When using ORDER BY xxx DESC, add NULLS LAST to exclude null records: ORDER BY xxx DESC NULLS LAST.\n"

- String Matching:
    - Don't directly match strings if you are not convinced. Use LOWER for fuzzy queries: WHERE LOWER(str) LIKE LOWER('%target_str%'). For example, to match 'meat lovers', use LOWER(str) LIKE '%meat%lovers%'.
    - For string-matching scenarios, convert non-standard symbols to '%'. e.g. ('he's to he%s)

- Date Handling:
    - For time-related queries, given the variety of formats, avoid using time converting functions unless you are certain of the specific format being used.

- Performance Tips:
    - Materialize complex expressions in CTEs to avoid recomputation.
    - You must quote all table names and column names in double quotes.
    - Filter early using WHERE clauses before applying aggregations.
"""
