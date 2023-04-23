name=aii_admin_backend
tag=chat-backend_test1
run:
	docker run --network host --name $(name) -v /etc/aii/aii_admin-backend.env:/app/aii_admin/.env -d  $(name):$(tag)
run_prod:
	docker run --restart always --network host --name dashboards-backend  -v /var/log/dashboards-backend/:/app/log -v /etc/ringme/dashboards-backend.env:/app/.env -d  $(name):$(tag)
build:
	docker build -t $(name):$(tag) .
stop:
	docker stop $(name)
rm:
	docker rm $(name)
push:
	docker tag $(name):$(tag) dextr/$(name):$(tag) && docker push dextr/$(name):$(tag)
update:
	make build && make push
rerun:
	make build && make stop && make rm && make run