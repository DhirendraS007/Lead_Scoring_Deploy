#Define Docker Python runtime
FROM public.ecr.aws/lambda/python:3.8

#Load Model file
COPY model/handler.py ${LAMBDA_TASK_ROOT}

# Install Dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN  pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Run Model File
CMD ["handler.lambda_handler"]